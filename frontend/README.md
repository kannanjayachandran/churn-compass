# Churn Compass Frontend

This is the production-grade frontend for Churn Compass, built with React, Vite, and shadcn/ui.

## Prerequisites

- Bun
- Backend API running at `http://localhost:8000`

## Setup

1. Install dependencies:
   ```bash
   bun install
   ```

2. Start the development server:
   ```bash
   bun run dev
   ```

3. Open `http://localhost:5173` in your browser.

## Configuration

The API URL can be configured in `.env` (create if needed):
```
VITE_API_URL=http://localhost:8000
```
Default is `http://localhost:8000`.

## Architecture

- **Framework**: React + Vite + TypeScript
- **Styling**: Tailwind CSS + shadcn/ui
- **State**: React Hooks (Local state simplicity)
- **Forms**: React Hook Form + Zod Validation
- **Charts**: Recharts (for SHAP explanations)
- **Icons**: Lucide React

## Features

- **Single Prediction**: form-based input with real-time risk scoring.
- **Explainability**: SHAP value visualization for every prediction.
- **Batch Processing**: CSV upload/download for bulk scoring.
- **Top-K Analysis**: Instant identification of high-risk customers from a batch.
