import React, { useState } from 'react';
import { parseCsv } from '@/lib/csvParser';
import { api } from '@/api/client';
import { TopKResponse, CustomerInput } from '@/api/types';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { Loader2 } from 'lucide-react';

export const TopKView: React.FC = () => {
    const [file, setFile] = useState<File | null>(null);
    const [result, setResult] = useState<TopKResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files?.[0]) {
            setFile(e.target.files[0]);
            setErrorMessage(null);
        }
    }

    const loadTopK = async () => {
        if (!file) return;
        setIsLoading(true);
        setErrorMessage(null);
        try {
            const text = await file.text();
            const rawCustomers = parseCsv(text);

            // Map CSV columns to API schema
            const customers = rawCustomers.map((c: any) => ({
                ...c,
                CardType: c["Card Type"] || c.CardType || c.Card_Type,
            }));

            // Limit for demo/safety ? No, backend limits batch.
            // But we assume the CSV matches the schema perfectly.

            const response = await api.getTopK(
                { customers: customers as CustomerInput[] },
                { k: 10 } // Default to top 10
            );
            setResult(response);
        } catch (e: any) {
            console.error("Top K Error", e);
            const detail = e.response?.data?.detail;

            let displayMsg = "Failed to analyze top k";

            if (typeof detail === 'string') {
                displayMsg = detail;
            } else if (Array.isArray(detail)) {
                // Handle Pydantic validation errors nicely
                // detail is [{loc:..., msg:..., type:..., input:...}]
                // We just want the message, not the input data which causes leaks
                displayMsg = detail.map(err => err.msg).join('; ');
            } else if (detail) {
                displayMsg = JSON.stringify(detail);
            }

            setErrorMessage(displayMsg);
        } finally {
            setIsLoading(false);
        }
    }

    return (
        <Card className="w-full shadow-sm border-border">
            <CardHeader>
                <CardTitle>High Risk Monitor</CardTitle>
                <CardDescription>Identify the Top-10 highest risk customers from your batch.</CardDescription>
            </CardHeader>
            <CardContent>
                {!result ? (
                    <div className="space-y-4">
                        <input type="file" accept=".csv" onChange={handleFile} className="block w-full text-sm text-slate-500
                            file:mr-4 file:py-2 file:px-4
                            file:rounded-full file:border-0
                            file:text-sm file:font-semibold
                            file:bg-violet-50 file:text-violet-700
                            hover:file:bg-violet-100
                        "/>
                        {errorMessage && (
                            <div className="p-3 text-sm text-red-500 bg-red-50 rounded-md border border-red-100">
                                {errorMessage}
                            </div>
                        )}
                        <Button onClick={loadTopK} disabled={!file || isLoading}>
                            {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                            Scan for High Risk
                        </Button>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <div className="flex justify-between items-center">
                            <h4 className="font-semibold">Top {result.k} At-Risk Customers</h4>
                            <Button variant="outline" size="sm" onClick={() => setResult(null)}>Reset</Button>
                        </div>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>Customer ID</TableHead>
                                    <TableHead>Risk Score</TableHead>
                                    <TableHead>Balance</TableHead>
                                    <TableHead>Country</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {result.customers.map((c, i) => (
                                    <TableRow key={i}>
                                        <TableCell className="font-mono text-xs">{c.CustomerId || "N/A"}</TableCell>
                                        <TableCell className="font-bold text-red-600">
                                            {(c.probability * 100).toFixed(1)}%
                                        </TableCell>
                                        <TableCell>{c.Balance}</TableCell>
                                        <TableCell>{c.Geography}</TableCell>
                                    </TableRow>
                                ))}
                            </TableBody>
                        </Table>
                    </div>
                )}
            </CardContent>
        </Card>
    )
}
