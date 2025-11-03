from dataclasses import dataclass, field
import uuid

# Class Definitions

@dataclass
class Applicant:
    name: str
    credit_score: int
    ss_num: str
    id: uuid.UUID = field(default_factory=uuid.uuid4) 

@dataclass
class ApplicantAnalysisResult:
    applicant: Applicant
    total_score: int # 0-100
    public_record_score: int # 0-100
    financial_records_score: int # 0-100

# Dummy demo data

applicants = [
    Applicant("Pablo", 600, "012-34-5678"),
    Applicant("Chris", 800, "123-45-6789"),
    Applicant("Daniel", 750, "234-56-7890"),
    Applicant("Jan", 850, "345-67-8901"),
]