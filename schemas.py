"""
Pydantic Schemas for FastAPI Fraud Detection Endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union


class TransactionInput(BaseModel):
    TransactionID: Optional[Union[int, str]] = Field(default=None, description="Unique transaction ID")
    TransactionDT: Optional[int] = Field(default=86400, description="Seconds since dataset reference epoch")
    TransactionAmt: float = Field(default=100.0, description="Transaction Amount in USD")
    ProductCD: Optional[str] = Field(default="W", description="Product Category (W, C, R, H, S)")
    
    card1: Optional[int] = Field(default=13824, description="Card Issuer BIN / Account Identification")
    card2: Optional[float] = Field(default=111.0, description="Card Sub-code")
    card3: Optional[float] = Field(default=150.0, description="Card Country Code")
    card4: Optional[str] = Field(default="visa", description="Card Brand (visa, mastercard, discover, etc.)")
    card5: Optional[float] = Field(default=226.0, description="Card Category Code")
    card6: Optional[str] = Field(default="debit", description="Card Type (debit, credit, etc.)")
    
    addr1: Optional[float] = Field(default=299.0, description="Purchaser Billing Region / Zip")
    addr2: Optional[float] = Field(default=87.0, description="Purchaser Country Code")
    dist1: Optional[float] = Field(default=None, description="Billing to Delivery Distance")
    dist2: Optional[float] = Field(default=None, description="IP Location to Billing Distance")
    
    P_emaildomain: Optional[str] = Field(default="gmail.com", description="Purchaser Email Domain")
    R_emaildomain: Optional[str] = Field(default=None, description="Recipient Email Domain")
    
    # C-Features (Counts)
    C1: Optional[float] = Field(default=1.0)
    C2: Optional[float] = Field(default=1.0)
    C3: Optional[float] = Field(default=0.0)
    C4: Optional[float] = Field(default=0.0)
    C5: Optional[float] = Field(default=0.0)
    C6: Optional[float] = Field(default=1.0)
    C7: Optional[float] = Field(default=0.0)
    C8: Optional[float] = Field(default=0.0)
    C9: Optional[float] = Field(default=1.0)
    C10: Optional[float] = Field(default=0.0)
    C11: Optional[float] = Field(default=1.0)
    C12: Optional[float] = Field(default=0.0)
    C13: Optional[float] = Field(default=1.0)
    C14: Optional[float] = Field(default=1.0)
    
    # D-Features (Days)
    D1: Optional[float] = Field(default=14.0)
    D2: Optional[float] = Field(default=None)
    D3: Optional[float] = Field(default=None)
    D4: Optional[float] = Field(default=0.0)
    D10: Optional[float] = Field(default=0.0)
    D11: Optional[float] = Field(default=None)
    D15: Optional[float] = Field(default=14.0)
    
    # Match Flags
    M1: Optional[str] = Field(default="T")
    M2: Optional[str] = Field(default="T")
    M3: Optional[str] = Field(default="T")
    M4: Optional[str] = Field(default="M0")
    M5: Optional[str] = Field(default="F")
    M6: Optional[str] = Field(default="T")
    M7: Optional[str] = Field(default=None)
    M8: Optional[str] = Field(default=None)
    M9: Optional[str] = Field(default=None)
    
    # Identity & Device
    DeviceType: Optional[str] = Field(default=None)
    DeviceInfo: Optional[str] = Field(default=None)
    id_01: Optional[float] = Field(default=None)
    id_02: Optional[float] = Field(default=None)
    id_12: Optional[str] = Field(default=None)
    id_30: Optional[str] = Field(default=None)
    id_31: Optional[str] = Field(default=None)

    # Dynamic extra fields supported (V-features, etc.)
    class Config:
        extra = "allow"


class SHAPDriver(BaseModel):
    feature: str
    description: str
    shapValue: float
    featureValue: Optional[Any] = None
    impact: str


class SHAPExplanation(BaseModel):
    baseValue: float
    topDrivers: List[SHAPDriver]
    topRiskDrivers: List[SHAPDriver]
    topSafetyDrivers: List[SHAPDriver]
    narrativeExplanation: str


class PredictionResponse(BaseModel):
    transactionId: Union[int, str]
    fraudProbability: float
    prediction: int
    riskLevel: str
    recommendation: str
    threshold: float
    modelVersion: str
    shapExplanation: SHAPExplanation
    processingTimeMs: float


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionInput]


class BatchPredictionResponse(BaseModel):
    totalProcessed: int
    fraudCount: int
    legitimateCount: int
    averageFraudProbability: float
    predictions: List[PredictionResponse]
    processingTimeMs: float


class RetrainRequest(BaseModel):
    feedbackTransactions: Optional[List[Dict[str, Any]]] = Field(default=None)


class RetrainResponse(BaseModel):
    timestamp: str
    championVersion: str
    candidateVersion: str
    newFeedbackSamples: int
    totalTrainSamples: int
    championMetrics: Dict[str, Any]
    candidateMetrics: Dict[str, Any]
    rocAucDelta: float
    gatePassed: bool
    status: str


class ModelInfoResponse(BaseModel):
    version: str
    active: bool
    algorithm: str
    objective: str
    numFeatures: int
    trainSamples: int
    validationSamples: int
    bestIteration: int
    metrics: Dict[str, Any]
    lastRetrained: Optional[str] = None
