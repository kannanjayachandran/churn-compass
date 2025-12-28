import { useRef, useState } from "react";
import axios from "axios";
import {
    UploadCloud,
    FileType,
    CheckCircle,
    AlertCircle,
    Loader2,
    ArrowLeft,
    BarChart3,
    Users,
    AlertTriangle,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { api } from "@/api/client";
import { parseCsv } from "@/lib/csvParser";
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Cell,
} from "recharts";



type CsvPredictionRow = {
    risk_level: "high" | "medium" | "low";
    [key: string]: unknown;
};



export function UploadForm() {
    const [file, setFile] = useState<File | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [status, setStatus] = useState<"idle" | "success" | "error">("idle");
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [results, setResults] = useState<CsvPredictionRow[] | null>(null);

    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setStatus("idle");
            setErrorMessage(null);
        }
    };

    const handleUpload = async () => {
        if (!file) return;

        setIsLoading(true);
        setErrorMessage(null);

        try {
            const blob = await api.predictCsv(file);

            // 1. Trigger download
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `predictions_${new Date().toISOString()}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);

            // 2. Parse for visualization
            const text = await blob.text();
            const parsedData = parseCsv(text) as CsvPredictionRow[];
            setResults(parsedData);

            setStatus("success");
        } catch (error: unknown) {
            console.error(error);
            let msg = "Upload failed. Check format.";

            if (axios.isAxiosError(error) && error.response?.data instanceof Blob) {
                try {
                    const text = await error.response.data.text();
                    const json = JSON.parse(text);
                    msg = json.detail || msg;
                } catch (e) {
                    console.error("Failed to parse error blob", e);
                }
            } else if (
                typeof error === "object" &&
                error !== null &&
                "response" in error
            ) {
                msg = (error as any).response?.data?.detail || msg;
            }

            setErrorMessage(msg || "Upload failed. Check format.");
            setStatus("error");
        } finally {
            setIsLoading(false);
        }
    };

    const resetView = () => {
        setResults(null);
        setFile(null);
        setStatus("idle");
        setErrorMessage(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    /* Results View */

    if (results) {
        const total = results.length;
        const highRisk = results.filter(r => r.risk_level === "high").length;
        const mediumRisk = results.filter(r => r.risk_level === "medium").length;
        const lowRisk = results.filter(r => r.risk_level === "low").length;

        const chartData = [
            { name: "Low Risk", count: lowRisk, color: "#22c55e" },
            { name: "Medium Risk", count: mediumRisk, color: "#eab308" },
            { name: "High Risk", count: highRisk, color: "#ef4444" },
        ];

        return (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="flex items-center justify-between">
                    <div>
                        <h2 className="text-2xl font-bold tracking-tight">
                            Batch Analysis Complete
                        </h2>
                        <p className="text-muted-foreground">
                            Scored {total} customers from {file?.name}
                        </p>
                    </div>
                    <Button variant="outline" onClick={resetView}>
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Analyze Another File
                    </Button>
                </div>

                <div className="grid gap-4 md:grid-cols-3">
                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">
                                Total Scanned
                            </CardTitle>
                            <Users className="h-4 w-4 text-muted-foreground" />
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">{total}</div>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">
                                High Risk Detected
                            </CardTitle>
                            <AlertTriangle className="h-4 w-4 text-red-500" />
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold text-red-600">
                                {highRisk}
                            </div>
                            <p className="text-xs text-muted-foreground">
                                {((highRisk / total) * 100).toFixed(1)}% of total
                            </p>
                        </CardContent>
                    </Card>

                    <Card>
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <CardTitle className="text-sm font-medium">
                                Action Required
                            </CardTitle>
                            <BarChart3 className="h-4 w-4 text-muted-foreground" />
                        </CardHeader>
                        <CardContent>
                            <div className="text-2xl font-bold">
                                {highRisk + mediumRisk}
                            </div>
                            <p className="text-xs text-muted-foreground">
                                Customers need attention
                            </p>
                        </CardContent>
                    </Card>
                </div>

                <Card>
                    <CardHeader>
                        <CardTitle>Risk Distribution</CardTitle>
                        <CardDescription>
                            Breakdown of customer base by churn risk level
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="h-100">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                <XAxis dataKey="name" axisLine={false} tickLine={false} />
                                <YAxis axisLine={false} tickLine={false} />
                                <Tooltip />
                                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                                    {chartData.map((entry, index) => (
                                        <Cell key={index} fill={entry.color} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            </div>
        );
    }

    /* ------------------------------------------------------------------ */
    /* Upload View */
    /* ------------------------------------------------------------------ */

    return (
        <Card className="w-full h-full shadow-sm border-border flex flex-col justify-between">
            <CardHeader>
                <CardTitle>Batch Prediction</CardTitle>
                <CardDescription>
                    Upload a CSV file containing customer data to generate bulk predictions.
                </CardDescription>
            </CardHeader>

            <CardContent>
                <div
                    className="border-2 border-dashed rounded-lg p-10 flex flex-col items-center justify-center text-center hover:bg-muted/50 transition-colors cursor-pointer"
                    onClick={() => fileInputRef.current?.click()}
                >
                    <input
                        ref={fileInputRef}
                        type="file"
                        className="hidden"
                        accept=".csv"
                        onChange={handleFileChange}
                    />

                    {file ? (
                        <div className="flex flex-col items-center gap-2">
                            <FileType className="h-10 w-10 text-primary" />
                            <p className="font-medium">{file.name}</p>
                            <p className="text-xs text-muted-foreground">
                                {(file.size / 1024).toFixed(2)} KB
                            </p>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center gap-2">
                            <UploadCloud className="h-10 w-10 text-muted-foreground" />
                            <p className="font-medium text-muted-foreground">
                                Click to upload CSV
                            </p>
                            <p className="text-xs text-muted-foreground">Max 50MB</p>
                        </div>
                    )}
                </div>

                {status === "success" && (
                    <div className="mt-4 flex items-center gap-2 text-green-600 bg-green-50 p-3 rounded-md">
                        <CheckCircle className="h-5 w-5" />
                        <span className="text-sm font-medium">
                            Predictions downloaded successfully
                        </span>
                    </div>
                )}

                {status === "error" && (
                    <div className="mt-4 flex items-center gap-2 text-destructive bg-red-50 p-3 rounded-md">
                        <AlertCircle className="h-5 w-5" />
                        <span className="text-sm font-medium">{errorMessage}</span>
                    </div>
                )}
            </CardContent>

            <CardFooter>
                <Button
                    className="w-full"
                    onClick={handleUpload}
                    disabled={!file || isLoading}
                >
                    {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                    Generate Predictions
                </Button>
            </CardFooter>
        </Card>
    );
}
