import sys
import os
import random
import flyte
import math
import asyncio
from pathlib import Path

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', ".."))
sys.path.insert(0, root_dir)
root_path = Path(root_dir)

from v2.agents.demo_data import Applicant, applicants, ApplicantAnalysisResult


agent_env = flyte.TaskEnvironment(
    name="agent_env",
    resources=flyte.Resources(memory="1Gi"),
    image=flyte.Image.from_debian_base()\
        .with_pip_packages(
            "flyte", "unionai-reuse"
            ),
)

@agent_env.task()
async def retrieve_applicants() -> list[Applicant]:
    num_of_applicants = random.randint(1,4)
    return applicants[:num_of_applicants]

@agent_env.task()
async def agent_public_records(a: Applicant) -> int:
    return random.randint(50,99)

@agent_env.task()
async def agent_financial_records(a: Applicant) -> int:
    return random.randint(50,99)

@agent_env.task()
async def finalize_analysis(ar: ApplicantAnalysisResult) -> ApplicantAnalysisResult:
    total_score = (ar.public_record_score + ar.financial_records_score)/2
    ar.total_score = total_score
    return ar

@agent_env.task()
async def process_applicant(a: Applicant) -> ApplicantAnalysisResult:
    with flyte.group(f"Analyze Applicant: ({a.name})"):
        res = await asyncio.gather(
            agent_public_records(a),
            agent_financial_records(a)
        )
        ar = ApplicantAnalysisResult(a, 0, res[0], res[1])
    return await finalize_analysis(ar)

@agent_env.task()
async def application_recommendation(analysis_results: list[ApplicantAnalysisResult]) -> int:
    total_scores = [ar.total_score for ar in analysis_results]
    score_avg = int(sum(total_scores)/len(total_scores))
    return score_avg

@agent_env.task()
async def process_applicants():
    
    aps = await retrieve_applicants()
    tasks = [process_applicant(ap) for ap in aps]
    res = await asyncio.gather(*tasks)
    recommendation = await application_recommendation(res)
    
# asyncio.run(process_applicants())
    
if __name__ == "__main__":
    
    # Run remotely with remote config
    flyte.init_from_config("../config.yaml", root_dir=root_path)

    # Run locally
    # flyte.init()

    run = flyte.run(process_applicants)
    print(run.name)
    print(run.url)
    run.wait(run)