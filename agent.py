"""LangGraph agent for booking mock diagnostic-lab appointments."""

import json
import os
import sys
import uuid
from pathlib import Path
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt


BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "mock_data.json"
SYSTEM_PROMPT = """You help users find previous lab tests and make mock diagnostic-lab bookings.
Use tools to answer requests and never invent results. Call search_emails before
extract_lab_tests when the user asks about previous tests. For bookings, search
labs for the exact requested test, check availability at a supporting lab, choose
the earliest slot by its ISO date-time, book it, then send confirmation. Never
substitute another test. If the user asks to book a generic MRI without specifying
the type, ask which type they mean and do not call booking tools yet. Ask a concise
question when any requested test is too ambiguous. If the user declines booking approval, stop the booking workflow and
do not ask for approval again. Only report a new booking when book_test returns
success true. If book_test reports a duplicate, explain that no new booking was
created and report the existing booking; do not send another confirmation.
Only call send_confirmation after a new successful booking. Keep previous tests
distinct from the newly requested test. Finish with a clear summary."""


class BookingState(MessagesState):
    """Conversation messages plus the current approval decision."""
    approved_booking: bool


def read_data():
    """Load the mock records and saved booking state."""
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def write_data(data):
    """Save mock state between CLI runs."""
    DATA_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


@tool
def search_emails(query: str) -> dict:
    """Search mock emails and lab reports for information matching a query."""
    data = read_data()
    words = [word.lower() for word in query.split() if len(word) > 2]
    matches = [email for email in data["emails"]
               if "report" in email["type"].lower()
               or any(word in email["subject"].lower() + " " + email["body"].lower()
                      for word in words)]
    return {"reports": matches}


@tool
def extract_lab_tests(data: dict) -> dict:
    """Extract completed test names from reports returned by search_emails."""
    reports = data.get("reports", [])
    found = []
    for report in reports:
        text = (report.get("subject", "") + " " + report.get("body", "")).lower()
        for test in read_data()["completed_test_names"]:
            if test.lower() in text and test not in found:
                found.append(test)
    return {"previousTests": found}


@tool
def search_labs(test_name: str) -> dict:
    """Find diagnostic labs that support the exact requested test name."""
    data = read_data()
    labs = [lab for lab in data["labs"]
            if test_name.casefold() in [test.casefold() for test in lab["tests"]]]
    return {"testName": test_name, "labs": labs}


@tool
def check_availability(lab_id: str, test_name: str) -> dict:
    """Check mock appointment slots for an exact supported test at one lab."""
    data = read_data()
    lab = next((item for item in data["labs"] if item["id"] == lab_id), None)
    canonical_test_name = next(
        (test for test in lab["tests"] if test.casefold() == test_name.casefold()),
        None,
    ) if lab else None
    if not lab or canonical_test_name is None:
        return {"slots": [], "error": "Lab does not support the exact requested test."}
    slots = lab["slots"].get(canonical_test_name, [])
    data["availability_checks"].append(
        {"labId": lab_id, "testName": canonical_test_name, "slots": slots})
    write_data(data)
    return {"testName": canonical_test_name, "slots": slots}


@tool
def book_test(lab_id: str, test_name: str, slot: str) -> dict:
    """Book the earliest checked slot for an exact supported test, if not duplicated."""
    data = read_data()
    lab = next((item for item in data["labs"] if item["id"] == lab_id), None)
    if not lab or test_name.casefold() not in [test.casefold() for test in lab["tests"]]:
        return {"success": False, "error": "Lab does not support the exact requested test."}
    checks = [check for check in data["availability_checks"]
              if check["labId"] == lab_id
              and check["testName"].casefold() == test_name.casefold()]
    if not checks or not checks[-1]["slots"]:
        return {"success": False, "error": "Check availability before booking."}
    earliest = min(checks[-1]["slots"])
    if slot != earliest:
        return {"success": False, "error": "Only the earliest available slot can be booked.",
                "earliestSlot": earliest}
    duplicate = next((booking for booking in data["bookings"]
                     if booking["testName"].casefold() == test_name.casefold()
                     and booking["labId"] == lab_id), None)
    if duplicate:
        return {"success": False, "duplicate": True,
                "error": "This test is already booked.", "booking": duplicate}
    booking = {"bookingId": f"LAB-{1000 + len(data['bookings']) + 1}",
               "testName": test_name, "labId": lab_id,
               "labName": lab["name"], "slot": slot, "status": "confirmed"}
    data["bookings"].append(booking)
    data["created_booking_ids_this_run"].append(booking["bookingId"])
    write_data(data)
    return {"success": True, "booking": booking}


@tool
def send_confirmation(booking_details: dict) -> dict:
    """Store a mock confirmation record for a successfully confirmed booking."""
    if not booking_details or booking_details.get("status") != "confirmed":
        return {"success": False, "error": "No confirmed booking to notify about."}
    data = read_data()
    if not any(item["bookingId"] == booking_details.get("bookingId")
               for item in data["bookings"]):
        return {"success": False, "error": "Booking was not found in mock data."}
    booking_id = booking_details["bookingId"]
    if booking_id not in data["created_booking_ids_this_run"]:
        return {"success": False,
                "error": "Only a new booking created during this request can be confirmed."}
    if booking_id in data["confirmed_booking_ids_this_run"]:
        return {"success": False, "error": "Confirmation was already recorded for this request."}
    data["confirmations"].append({"bookingId": booking_id,
                                  "message": "Mock confirmation recorded."})
    data["confirmed_booking_ids_this_run"].append(booking_id)
    write_data(data)
    return {"success": True, "message": "Mock confirmation recorded."}


TOOLS = [search_emails, extract_lab_tests, search_labs,
         check_availability, book_test, send_confirmation]


def create_agent():
    """Build a LangGraph agent, approval, and tool workflow."""
    model = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-5.5"),
        api_key=os.getenv("CODECRAFT_API_KEY"),
        base_url=os.getenv("BASE_URL", "https://codecraftapi.com/v1"),
        use_responses_api=False,
    )
    model_with_tools = model.bind_tools(TOOLS)

    def call_model(state: MessagesState):
        response = model_with_tools.invoke(
            [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    def approval_node(state: BookingState):
        """Pause for approval before allowing the pending booking tool to run."""
        pending_calls = state["messages"][-1].tool_calls
        booking_call = next(call for call in pending_calls
                            if call["name"] == "book_test")
        arguments = booking_call["args"]
        lab = next((item for item in read_data()["labs"]
                    if item["id"] == arguments["lab_id"]), None)
        decision = interrupt({
            "action": "booking_approval",
            "test": arguments["test_name"],
            "lab": lab["name"] if lab else arguments["lab_id"],
            "appointment": arguments["slot"],
            "question": "Approve this mock booking? Enter yes or no.",
        })
        approved = isinstance(decision, dict) and decision.get("approved") is True
        if approved:
            return {"approved_booking": True}
        rejected_messages = [
            ToolMessage(
                content=json.dumps({"success": False, "approved": False,
                                    "error": "The user declined the booking."}),
                tool_call_id=call["id"], name=call["name"])
            for call in pending_calls
        ]
        return {"messages": rejected_messages, "approved_booking": False}

    def route_booking_approval(state: BookingState):
        """Run tools only after approval; after decline, return to the agent."""
        return "tools" if state.get("approved_booking") else "agent"

    def route_agent(state: BookingState):
        """Pause for approval only when the requested booking is valid and new."""
        last_message = state["messages"][-1]
        for call in last_message.tool_calls:
            if call["name"] == "book_test" and booking_needs_approval(call["args"]):
                return "approval"
        return tools_condition(state)

    graph = StateGraph(BookingState)
    graph.add_node("agent", call_model)
    graph.add_node("approval", approval_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_agent,
                                {"approval": "approval", "tools": "tools", END: END})
    graph.add_conditional_edges("approval", route_booking_approval,
                                {"tools": "tools", "agent": "agent"})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=InMemorySaver())


def load_local_environment():
    """Read simple KEY=VALUE settings from .env without another dependency."""
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                os.environ.setdefault(name.strip(), value.strip())


def booking_needs_approval(arguments: dict) -> bool:
    """Require approval only for a supported, earliest, not-yet-booked slot."""
    data = read_data()
    test_name = arguments["test_name"]
    lab = next((item for item in data["labs"] if item["id"] == arguments["lab_id"]), None)
    if not lab or test_name.casefold() not in [test.casefold() for test in lab["tests"]]:
        return False
    checks = [check for check in data["availability_checks"]
              if check["labId"] == arguments["lab_id"]
              and check["testName"].casefold() == test_name.casefold()]
    if not checks or not checks[-1]["slots"]:
        return False
    if arguments["slot"] != min(checks[-1]["slots"]):
        return False
    return not any(booking["labId"] == arguments["lab_id"]
                   and booking["testName"].casefold() == test_name.casefold()
                   for booking in data["bookings"])


def run_agent(request: str) -> str:
    """Run the graph and return the final assistant message."""
    data = read_data()
    data["availability_checks"] = []
    data["created_booking_ids_this_run"] = []
    data["confirmed_booking_ids_this_run"] = []
    write_data(data)
    graph = create_agent()
    config = {"configurable": {"thread_id": str(uuid.uuid4())},
              "recursion_limit": 30}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": request}],
         "approved_booking": False}, config=config)
    if result.get("__interrupt__"):
        approval_request = result["__interrupt__"][0].value
        print("\nBooking approval requested")
        print(f"  Test: {approval_request['test']}")
        print(f"  Lab: {approval_request['lab']}")
        print(f"  Appointment: {approval_request['appointment']}")
        answer = input(f"{approval_request['question']} ").strip().lower()
        approved = answer in {"y", "yes"}
        result = graph.invoke(Command(resume={"approved": approved}), config=config)
    return result["messages"][-1].content


def main():
    """Load settings, read the user's request, and print the agent response."""
    load_local_environment()
    request = " ".join(sys.argv[1:]).strip()
    if not request:
        request = input("What would you like me to do? ").strip()
    if not os.getenv("CODECRAFT_API_KEY"):
        print("Set CODECRAFT_API_KEY in .env before running the agent.")
        raise SystemExit(1)
    print(run_agent(request))


if __name__ == "__main__":
    main()
