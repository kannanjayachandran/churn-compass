import axios from "axios";
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
} from "./types";

/* ------------------------------------------------------------------ */
/* Config */
/* ------------------------------------------------------------------ */

const API_URL = import.meta.env.VITE_API_URL;

if (!API_URL) {
    throw new Error("VITE_API_URL is not defined");
}

const client = axios.create({
    baseURL: API_URL,
    headers: {
        "Content-Type": "application/json",
    },
});

/* API */

export const api = {
    getHealth: async (): Promise<HealthResponse> => {
        const response = await client.get<HealthResponse>("/health");
        return response.data;
    },

    getVersion: async (): Promise<VersionResponse> => {
        const response = await client.get<VersionResponse>("/version");
        return response.data;
    },

    getSystemStatus: async (): Promise<SystemStatusResponse> => {
        const response = await client.get<SystemStatusResponse>("/system/status");
        return response.data;
    },

    predictSingle: async (
        data: CustomerInput
    ): Promise<PredictionResponse> => {
        const response = await client.post<PredictionResponse>("/predict", data);
        return response.data;
    },

    explainPrediction: async (
        data: CustomerInput,
        topN: number = 10
    ): Promise<ExplanationResponse> => {
        const response = await client.post<ExplanationResponse>(
            "/explain",
            data,
            { params: { top_n: topN } }
        );
        return response.data;
    },

    predictBatch: async (
        data: BatchPredictionRequest
    ): Promise<BatchPredictionResponse> => {
        const response = await client.post<BatchPredictionResponse>(
            "/predict_batch",
            data
        );
        return response.data;
    },

    predictCsv: async (file: File): Promise<Blob> => {
        const formData = new FormData();
        formData.append("file", file);

        const response = await client.post("/predict_csv", formData, {
            responseType: "blob",
            headers: {
                "Content-Type": "multipart/form-data",
            },
        });

        return response.data;
    },

    getTopK: async (
        request: BatchPredictionRequest,
        params: TopKRequest
    ): Promise<TopKResponse> => {
        const response = await client.post<TopKResponse>("/top_k", {
            request,
            params,
        });
        return response.data;
    },
};
