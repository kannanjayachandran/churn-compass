import axios from 'axios';
import {
    CustomerInput,
    PredictionResponse,
    ExplanationResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    TopKRequest,
    TopKResponse,
    HealthResponse,
    VersionResponse,
    SystemStatusResponse,
} from './types';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const client = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const api = {
    getHealth: async (): Promise<HealthResponse> => {
        const response = await client.get<HealthResponse>('/health');
        return response.data;
    },

    getVersion: async (): Promise<VersionResponse> => {
        const response = await client.get<VersionResponse>('/version');
        return response.data;
    },

    getSystemStatus: async (): Promise<SystemStatusResponse> => {
        const response = await client.get<SystemStatusResponse>('/system/status');
        return response.data;
    },

    predictSingle: async (data: CustomerInput): Promise<PredictionResponse> => {
        const response = await client.post<PredictionResponse>('/predict', data);
        return response.data;
    },

    explainPrediction: async (data: CustomerInput, topN: number = 10): Promise<ExplanationResponse> => {
        // Note: top_n is query param in the py code? No, let's check.
        // @app.post("/explain") async def explain_prediction(customer: CustomerInput, top_n: int = 10, ...)
        // Usually scalar args become Query params in FastAPI if body is Pydantic model.
        const response = await client.post<ExplanationResponse>('/explain', data, {
            params: { top_n: topN },
        });
        return response.data;
    },

    predictBatch: async (data: BatchPredictionRequest): Promise<BatchPredictionResponse> => {
        const response = await client.post<BatchPredictionResponse>('/predict_batch', data);
        return response.data;
    },

    predictCsv: async (file: File): Promise<Blob> => {
        const formData = new FormData();
        formData.append('file', file);
        const response = await client.post('/predict_csv', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            responseType: 'blob',
        });
        return response.data;
    },

    getTopK: async (batchRequest: BatchPredictionRequest, topKParams: TopKRequest): Promise<TopKResponse> => {
        // Based on FastAPI behavior for multiple Body models:
        const response = await client.post<TopKResponse>('/top_k', {
            request: batchRequest,
            params: topKParams
        });
        return response.data;
    },
};

export default api;
