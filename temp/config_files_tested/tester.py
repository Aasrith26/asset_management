import requests


def download_job_files(job_id, base_url="http://localhost:8000"):
    """Download all files for a specific job"""

    # Download CSV
    csv_url = f"{base_url}/jobs/{job_id}/csv?download=true"
    csv_response = requests.get(csv_url)
    with open(f"assets_data_{job_id}.csv", "wb") as f:
        f.write(csv_response.content)

    # Download context files
    assets = ['NIFTY50', 'GOLD', 'BITCOIN', 'REIT']
    for asset in assets:
        context_url = f"{base_url}/jobs/{job_id}/context/{asset}?download=true"
        context_response = requests.get(context_url)
        with open(f"{asset}_context_{job_id}.json", "wb") as f:
            f.write(context_response.content)

    print(f"Downloaded all files for job {job_id}")


# Usage
download_job_files("ce8a49e9-eefc-4287-82bf-59e1b62e686f")
