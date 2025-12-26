import React from 'react';
import { PredictionResponse, ExplanationResponse } from '@/api/types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { cn } from '@/lib/utils';
import { AlertCircle, CheckCircle2 } from 'lucide-react';

interface ResultsDisplayProps {
    prediction: PredictionResponse | null;
    explanation: ExplanationResponse | null;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ prediction, explanation }) => {
    if (!prediction) return null;

    const riskColor = prediction.risk_level === 'high' ? 'text-red-600' :
        prediction.risk_level === 'medium' ? 'text-amber-600' : 'text-green-600';

    const riskBg = prediction.risk_level === 'high' ? 'bg-red-50 border-red-200' :
        prediction.risk_level === 'medium' ? 'bg-amber-50 border-amber-200' : 'bg-green-50 border-green-200';

    // Prepare chart data
    const chartData = explanation?.explanation.top_features.map(f => ({
        name: f.feature,
        value: Math.abs(f.contribution),
        impact: f.impact,
        originalValue: f.contribution
    })) || [];

    return (
        <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <Card className={cn("border-l-4", riskBg.replace('bg-', 'border-l-'))}>
                <CardHeader className="pb-2">
                    <CardTitle className="text-lg font-medium text-muted-foreground">Churn Risk Assessment</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="flex items-center gap-4">
                        {prediction.risk_level === 'high' ? (
                            <AlertCircle className="h-12 w-12 text-red-600" />
                        ) : (
                            <CheckCircle2 className="h-12 w-12 text-green-600" />
                        )}
                        <div>
                            <div className={cn("text-4xl font-bold tracking-tight", riskColor)}>
                                {prediction.risk_level.toUpperCase()}
                            </div>
                            <div className="text-sm text-muted-foreground">
                                Probability of churn: <span className="font-medium text-foreground">{(prediction.probability * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>

                    <div className="mt-4 pt-4 border-t">
                        <p className="text-sm text-muted-foreground">
                            Action Recommendation:
                            {prediction.risk_level === 'high' ? (
                                <span className="font-medium text-foreground ml-1">Immediate intervention required. Offer retention incentives.</span>
                            ) : prediction.risk_level === 'medium' ? (
                                <span className="font-medium text-foreground ml-1">Monitor closely. Send satisfaction survey.</span>
                            ) : (
                                <span className="font-medium text-foreground ml-1">No immediate action needed.</span>
                            )}
                        </p>
                    </div>
                </CardContent>
            </Card>

            {explanation && (
                <Card>
                    <CardHeader>
                        <CardTitle>Key Drivers (SHAP)</CardTitle>
                        <CardDescription>
                            Factors contributing most to this prediction.
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart layout="vertical" data={chartData} margin={{ left: 20 }}>
                                <XAxis type="number" hide />
                                <YAxis dataKey="name" type="category" width={100} tick={{ fontSize: 12 }} />
                                <Tooltip
                                    formatter={(_value: number, _name: string, props: any) => [
                                        props.payload.originalValue.toFixed(4),
                                        "Contribution"
                                    ]}
                                />
                                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                    {chartData.map((entry, index) => (
                                        <Cell
                                            key={`cell-${index}`}
                                            fill={entry.impact === 'increase' ? '#ef4444' : '#22c55e'}
                                        />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                        <div className="flex justify-center gap-6 text-xs text-muted-foreground mt-2">
                            <div className="flex items-center gap-1">
                                <div className="w-3 h-3 bg-red-500 rounded-sm"></div>
                                <span>Increases Risk</span>
                            </div>
                            <div className="flex items-center gap-1">
                                <div className="w-3 h-3 bg-green-500 rounded-sm"></div>
                                <span>Decreases Risk</span>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
};
