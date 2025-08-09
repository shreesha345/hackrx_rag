"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, HttpUrl
from typing import List, Union, Optional

class DocumentRequest(BaseModel):
    documents: Union[str, HttpUrl]
    questions: List[str]
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                    "questions": [
                        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                        "What is the waiting period for pre-existing diseases (PED) to be covered?",
                        "Does this policy cover maternity expenses, and what are the conditions?",
                        "What is the waiting period for cataract surgery?",
                        "Are the medical expenses for an organ donor covered under this policy?",
                        "What is the No Claim Discount (NCD) offered in this policy?",
                        "Is there a benefit for preventive health check-ups?",
                        "How does the policy define a 'Hospital'?",
                        "What is the extent of coverage for AYUSH treatments?",
                        "Are there any sub-limits on room rent and ICU charges for Plan A?"
                    ]
                },
                {
                    "documents": "https://example.com/email.eml",
                    "questions": [
                        "What is the subject of this email?",
                        "Who sent this email?",
                        "What are the main points discussed in this email?",
                        "Are there any attachments mentioned?"
                    ]
                },
                {
                    "documents": "https://example.com/outlook_email.msg",
                    "questions": [
                        "What is the subject of this email?",
                        "Who is the sender?",
                        "What is the content of this email?",
                        "When was this email sent?"
                    ]
                }
            ]
        }
    }

class DocumentResponse(BaseModel):
    answers: List[str]
    method: str
    processing_time: float
    rag_processing: str
    source: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "answers": [
                    "A grace period of thirty days is provided for premium payment after the due date",
                    "There is a waiting period of thirty-six months of continuous coverage for pre-existing diseases to be covered",
                    "Yes, this policy covers maternity expenses. The female Insured Person should have been continuously covered for at least 24 months before availing this benefit. Coverage is not available for female Insured Persons below eighteen years and above forty-five years of age. Delivery or termination within a waiting period of twenty-four months is also excluded, except when caused by an accident. The policy limits coverage to two deliveries or terminations. More than one delivery or termination during a single Policy Period is excluded. Maternity expenses of a Surrogate Mother are excluded unless the claim is admitted under Section 3.1.15 Infertility",
                    "There is a two-year waiting period for cataract surgery",
                    "Yes, the policy covers medical expenses for an organ donor's hospitalization for organ harvesting, provided the organ donation conforms to the Transplantation of Human Organs Act 1994, the organ is used for an Insured Person who has been medically advised to undergo an organ transplant, the medical expenses are incurred for the organ donor as an in-patient in a Hospital, and a claim has been admitted under the In-patient Treatment Section for the Insured Person undergoing the organ transplant",
                    "A No Claim Discount of a flat 5% is offered on the base premium on renewal of policies with a term of one year, provided no claims were reported in the expiring Policy. For policies with a term exceeding one year, the NCD amount for each claim-free policy year is aggregated and allowed on renewal, up to a maximum of 5% of the total base premium",
                    "Yes, expenses for a health check-up are reimbursed at the end of a block of two continuous policy years, provided the Policy has been continuously renewed without a break. Expenses are subject to the limit stated in the Table of Benefits",
                    "A Hospital means any institution established for in-patient care and day care treatment of disease/injuries and which has been registered as a hospital with the local authorities under the Clinical Establishments (Registration and Regulation) Act, 2010 or under the enactments specified under Schedule of Section 56(1) of the said Act, OR complies with all minimum criteria including having qualified nursing staff under its employment round the clock, at least ten inpatient beds in towns with a population of less than ten lacs and fifteen inpatient beds in all other places, qualified medical practitioner(s) in charge round the clock, a fully equipped operation theatre of its own where surgical procedures are carried out, and maintaining daily records of patients accessible to the Company's authorized personnel",
                    "The policy covers medical expenses incurred for Inpatient Care treatment under Ayurveda, Yoga and Naturopathy, Unani, Siddha and Homeopathy systems of medicines during each Policy Period up to the limit of Sum Insured as specified in the Policy Schedule in any AYUSH Hospital",
                    "For Plan A, room charges and intensive care unit charges per day are payable up to the limit shown in the Table of Benefits. However, the limit does not apply if the treatment is undergone for a listed procedure in a Preferred Provider Network (PPN) as a package"
                ],
                "method": "context_cache",
                "processing_time": 2.5,
                "rag_processing": "in_progress",
                "source": "gemini_cache"
            }
        }
    }

class ErrorResponse(BaseModel):
    error: str
    detail: str = None
    
class StatusResponse(BaseModel):
    document_id: str
    rag_status: str
    cache_protected: bool
    processing_time: float = None

class EmbeddingRequest(BaseModel):
    document_urls: List[Union[str, HttpUrl]]
    use_async: bool = True
    num_workers: int = 4
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "document_urls": [
                        "https://example.com/document1.pdf",
                        "https://example.com/presentation.pptx", 
                        "https://example.com/spreadsheet.xlsx",
                        "https://example.com/report.docx"
                    ],
                    "use_async": True,
                    "num_workers": 4
                },
                {
                    "document_urls": [
                        "https://storage.com/policy.pdf",
                        "https://docs.com/manual.pdf"
                    ],
                    "use_async": False,
                    "num_workers": 2
                }
            ]
        }
    }

class DocumentProcessingResult(BaseModel):
    url: str
    status: str  # "success", "failed", "skipped"
    message: str
    processing_time: float = None
    file_type: str = None

class EmbeddingResponse(BaseModel):
    total_documents: int
    processed_successfully: int
    failed_documents: int
    skipped_documents: int
    processing_time: float
    results: List[DocumentProcessingResult]
    use_async: bool
    num_workers: int
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "total_documents": 4,
                "processed_successfully": 3,
                "failed_documents": 1,
                "skipped_documents": 0,
                "processing_time": 45.2,
                "use_async": True,
                "num_workers": 4,
                "results": [
                    {
                        "url": "https://example.com/document1.pdf",
                        "status": "success",
                        "message": "Document processed and indexed successfully",
                        "processing_time": 12.5,
                        "file_type": "document"
                    },
                    {
                        "url": "https://example.com/presentation.pptx",
                        "status": "success", 
                        "message": "Presentation processed via RAG pipeline",
                        "processing_time": 18.3,
                        "file_type": "presentation"
                    },
                    {
                        "url": "https://example.com/spreadsheet.xlsx",
                        "status": "success",
                        "message": "Spreadsheet processed via RAG pipeline", 
                        "processing_time": 14.4,
                        "file_type": "spreadsheet"
                    },
                    {
                        "url": "https://example.com/report.docx",
                        "status": "failed",
                        "message": "Failed to download document: URL not accessible",
                        "processing_time": 0.0,
                        "file_type": "document"
                    }
                ]
            }
        }
    }

class URLBlockerStatus(BaseModel):
    url_blocker_enabled: bool
    blocked_urls_count: int
    blocked_urls: Optional[List[str]] = None
    message: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "url_blocker_enabled": True,
                "blocked_urls_count": 3,
                "blocked_urls": [
                    "https://register.hackrx.in/utils/get-secret-token",
                    "https://register.hackrx.in/utils/get-secret-token?hackTeam=5693",
                    "register.hackrx.in/utils/get-secret-token"
                ],
                "message": "URL blocker is enabled and active"
            }
        }
    }

class URLBlockerToggle(BaseModel):
    url_blocker_enabled: bool
    blocked_urls_count: int
    message: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "url_blocker_enabled": True,
                "blocked_urls_count": 3,
                "message": "URL blocker has been enabled"
            }
        }
    }

class URLBlockerModify(BaseModel):
    url_blocker_enabled: bool
    blocked_urls_count: int
    added_url: Optional[str] = None
    removed_url: Optional[str] = None
    message: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "url_blocker_enabled": True,
                "blocked_urls_count": 4,
                "added_url": "https://example.com/blocked-url",
                "message": "URL has been added to the blocked list"
            }
        }
    }
