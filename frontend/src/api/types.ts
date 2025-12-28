/* Core Inputs */

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
    CardType: "Silver" | "Gold" | "Diamond" | "Platinum";
}

/* Prediction */

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

/* ------------------------------------------------------------------ */
/* Batch */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/* Top-K */
/* ------------------------------------------------------------------ */

export interface TopKRequest {
    k?: number;
    k_percent?: number;
}

export interface TopKCustomer {
    CustomerId?: string | number;
    probability: number;
    Balance?: number;
    Geography?: string;
    [key: string]: unknown;
}

export interface TopKResponse {
    customers: TopKCustomer[];
    k: number;
    k_percent: number;
}

/* ------------------------------------------------------------------ */
/* System / Health */
/* ------------------------------------------------------------------ */

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

export interface SystemInfo {
    platform: string;
    system?: string;
    memory_total_gb: number;
    memory_available_gb: number;
    cpu_percent: number;
    [key: string]: unknown;
}

export interface SystemStatusResponse extends HealthResponse {
    metrics: Record<string, number>;
    params: Record<string, unknown>;
    system_info: SystemInfo;
}
