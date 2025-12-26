export interface CustomerInput {
    CreditScore: number;
    Geography: "France" | "Spain" | "Germany";
    Gender: "Male" | "Female";
    Age: number;
    Tenure: number;
    Balance: number;
    NumOfProducts: number;
    HasCrCard: number; // 0 or 1
    IsActiveMember: number; // 0 or 1
    EstimatedSalary: number;
    CardType: "SILVER" | "GOLD" | "DIAMOND" | "PLATINUM";
}

export interface PredictionResponse {
    prediction: number;
    probability: number;
    risk_level: "low" | "medium" | "high";
}

export interface FeatureContribution {
    feature: string;
    contribution: number;
    impact: "increase" | "decrease";
}

export interface SHAPExplanation {
    top_features: FeatureContribution[];
    base_value: number;
    prediction_value: number;
}

export interface ExplanationResponse {
    prediction: number;
    probability: number;
    risk_level: "low" | "medium" | "high";
    explanation: SHAPExplanation;
}

export interface BatchPredictionRequest {
    customers: CustomerInput[];
    include_features?: boolean;
}

export interface BatchPredictionResponse {
    predictions: PredictionResponse[];
    summary: {
        total: number;
        high_risk: number;
        medium_risk: number;
        low_risk: number;
        mean_probability: number;
    };
}

export interface TopKRequest {
    k?: number;
    k_percent?: number;
}

export interface TopKResponse {
    customers: Array<Record<string, any>>;
    k: number;
    k_percent: number;
}

export interface HealthResponse {
    status: "healthy" | "unhealthy";
    model_loaded: boolean;
    timestamp: string;
    version: string;
}

export interface VersionResponse {
    api_version: string;
    model_name: string;
    model_version?: string;
    model_stage?: string;
    mlflow_tracking_uri: string;
}

export interface SystemStatusResponse extends HealthResponse {
    metrics: Record<string, number>;
    params: Record<string, any>;
    system_info: Record<string, any>;
}
