# Example requests

1. Successful booking: `Find my previous tests and book the earliest available Cardiac MRI.`
2. Unsupported test: `Book XYZ Advanced Scan.`
3. Missing information: `Book an MRI for me.`

The mock JSON has no lab that offers `XYZ Advanced Scan`. The generic MRI request asks which MRI test is intended. The successful request uses Cardiac MRI, supported only by Central Imaging Clinic in the mock data.
