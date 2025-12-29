import { PredictionResponse, ExplanationResponse } from '@/api/types';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';
import { cn } from '@/lib/utils';
import {
    AlertCircle,
    CheckCircle2,
    User,
    Calendar,
    Wallet,
    CreditCard,
    Package,
    Globe,
    TrendingUp,
    Activity,
    Coins,
    Award,
    Clock,
    UserCheck,
    Briefcase
} from 'lucide-react';

interface ResultsDisplayProps {
    prediction: PredictionResponse | null;
    explanation: ExplanationResponse | null;
}

const FeatureIconMap: Record<string, React.ReactNode> = {
    "Age": <Calendar className="h-3.5 w-3.5" />,
    "Balance": <Wallet className="h-3.5 w-3.5" />,
    "Credit Score": <Award className="h-3.5 w-3.5" />,
    "Num Of Products": <Package className="h-3.5 w-3.5" />,
    "Estimated Salary": <Coins className="h-3.5 w-3.5" />,
    "Tenure": <Clock className="h-3.5 w-3.5" />,
    "Has Credit Card": <CreditCard className="h-3.5 w-3.5" />,
    "Has Cr Card": <CreditCard className="h-3.5 w-3.5" />,
    "Is Active Member": <UserCheck className="h-3.5 w-3.5" />,
    "Geography Germany": <Globe className="h-3.5 w-3.5" />,
    "Geography Spain": <Globe className="h-3.5 w-3.5" />,
    "Geography France": <Globe className="h-3.5 w-3.5" />,
    "Gender Male": <User className="h-3.5 w-3.5" />,
    "Gender Female": <User className="h-3.5 w-3.5" />,
    "Balance Per Product": <Activity className="h-3.5 w-3.5" />,
    "Tenure Age Ratio": <TrendingUp className="h-3.5 w-3.5" />,
    "Is Zero Balance": <Activity className="h-3.5 w-3.5" />,
    "High Value Customer": <Briefcase className="h-3.5 w-3.5" />,
    "Age Group Young": <User className="h-3.5 w-3.5" />,
    "Age Group Middle": <User className="h-3.5 w-3.5" />,
    "Age Group Senior": <User className="h-3.5 w-3.5" />,
};

const CustomYAxisTick = (props: any) => {
    const { x, y, payload } = props;
    const icon = FeatureIconMap[payload.value] || <Activity className="h-3.5 w-3.5" />;

    return (
        <g transform={`translate(${x},${y})`}>
            {/* Extended width to ensure no clipping, lowered x to align with right edge of YAxis area */}
            <foreignObject x={-200} y={-10} width={180} height={20} overflow="visible">
                <div className="flex items-center justify-end gap-2 text-right">
                    <span className="text-[12px] font-medium text-slate-700 truncate max-w-[140px]">
                        {payload.value}
                    </span>
                    <span className="text-slate-400 shrink-0">
                        {icon}
                    </span>
                </div>
            </foreignObject>
        </g>
    );
};

// Action recommendations based on feature impact
const featureActionMap: Record<string, { increase: string; decrease: string }> = {
    "Num Of Products": {
        increase: "Review product bundling - customer may be over-committed",
        decrease: "Cross-sell additional products to increase stickiness"
    },
    "Age": {
        increase: "Target with age-appropriate retention offers",
        decrease: "Engage with youth-focused digital banking features"
    },
    "Is Active Member": {
        increase: "Customer engagement is low - activate re-engagement program",
        decrease: "Maintain current engagement level with loyalty rewards"
    },
    "Balance": {
        increase: "High balance risk - offer premium account benefits",
        decrease: "Incentivize deposits with competitive rates"
    },
    "Estimated Salary": {
        increase: "High earner at risk - provide personalized wealth services",
        decrease: "Offer value-tier products suited to income level"
    },
    "Geography Germany": {
        increase: "German market risk - review regional competitive offerings",
        decrease: "Leverage regional satisfaction drivers"
    },
    "Geography Spain": {
        increase: "Spanish market risk - address regional pain points",
        decrease: "Continue regional strategy success"
    },
    "Geography France": {
        increase: "French market risk - review local market conditions",
        decrease: "Maintain regional engagement approach"
    },
    "Tenure": {
        increase: "Long-term customer showing risk - priority VIP retention",
        decrease: "New customer - focus on onboarding experience"
    },
    "Has Credit Card": {
        increase: "Credit product driving risk - review terms and benefits",
        decrease: "Offer credit card with attractive rewards"
    },
    "Has Cr Card": {
        increase: "Credit product driving risk - review terms and benefits",
        decrease: "Offer credit card with attractive rewards"
    },
    "Balance Per Product": {
        increase: "Product-balance ratio unfavorable - optimize product mix",
        decrease: "Healthy product utilization - maintain current strategy"
    },
    "Gender Female": {
        increase: "Review offerings for female demographic preferences",
        decrease: "Continue current demographic targeting"
    },
    "Gender Male": {
        increase: "Review offerings for male demographic preferences",
        decrease: "Continue current demographic targeting"
    },
    "Is Zero Balance": {
        increase: "Zero balance indicates disengagement - initiate win-back",
        decrease: "Active account - maintain engagement"
    },
    "Credit Score": {
        increase: "Credit profile concern - offer credit improvement resources",
        decrease: "Strong credit customer - offer premium products"
    },
    "Tenure Age Ratio": {
        increase: "Loyalty ratio concerning - strengthen relationship",
        decrease: "Strong loyalty foundation - nurture with recognition"
    },
    "High Value Customer": {
        increase: "High-value at risk - assign dedicated relationship manager",
        decrease: "Develop customer value with growth incentives"
    },
    "Age Group Young": {
        increase: "Young customer at risk - enhance digital experience",
        decrease: "Engaged young customer - offer referral programs"
    },
    "Age Group Middle": {
        increase: "Mid-age customer risk - offer life-stage products",
        decrease: "Stable segment - maintain service quality"
    },
    "Age Group Senior": {
        increase: "Senior customer risk - simplify services and add support",
        decrease: "Loyal senior - recognize with tenure benefits"
    },
};

function generateDynamicRecommendations(
    topFeatures: Array<{ feature: string; contribution: number; impact: 'increase' | 'decrease' }> | undefined,
    riskLevel: 'high' | 'medium' | 'low'
): string[] {
    // For low risk, no recommendations needed
    if (riskLevel === 'low') {
        return ["No action needed - customer is in good standing."];
    }

    if (!topFeatures || topFeatures.length === 0) {
        // Fallback to static recommendations
        if (riskLevel === 'high') return ["Immediate intervention required. Offer retention incentives."];
        return ["Monitor closely. Send satisfaction survey."];
    }

    // Get top 3 features sorted by absolute contribution
    const sortedFeatures = [...topFeatures]
        .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
        .slice(0, 3);

    const recommendations: string[] = [];

    for (const feature of sortedFeatures) {
        const actions = featureActionMap[feature.feature];
        if (actions) {
            // Use the recommendation based on whether feature increases or decreases risk
            const rec = feature.impact === 'increase' ? actions.increase : actions.decrease;
            recommendations.push(rec);
        }
    }

    // Fallback if no mapped features found
    if (recommendations.length === 0) {
        if (riskLevel === 'high') return ["Immediate intervention required. Offer retention incentives."];
        if (riskLevel === 'medium') return ["Monitor closely. Send satisfaction survey."];
        return ["No immediate action needed."];
    }

    return recommendations;
}

export function ResultsDisplay({ prediction, explanation }: ResultsDisplayProps) {

    if (!prediction) return null;

    // Generate dynamic recommendations based on top drivers
    const dynamicRecommendations = generateDynamicRecommendations(
        explanation?.explanation.top_features,
        prediction.risk_level
    );

    // derive both class from same source
    const riskStyles = {
        high: {
            text: "text-red-600",
            bg: "bg-red-50 border-red-200",
            border: "border-l-red-500",
        },
        medium: {
            text: "text-amber-600",
            bg: "bg-amber-50 border-amber-200",
            border: "border-l-amber-500",
        },
        low: {
            text: "text-green-600",
            bg: "bg-green-50 border-green-200",
            border: "border-l-green-500",
        },
    } as const;

    const styles = riskStyles[prediction.risk_level];

    // Prepare chart data and calculate total impact for percentage
    const totalImpact = explanation?.explanation.top_features.reduce((acc, f) => acc + Math.abs(f.contribution), 0) || 1;

    const chartData = explanation?.explanation.top_features.map(f => ({
        name: f.feature,
        value: Math.abs(f.contribution),
        percentage: (f.contribution / totalImpact) * 100,
        impact: f.impact,
        originalValue: f.contribution
    })) || [];


    return (
        <div className="h-full flex flex-col space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
            <Card className={cn("border-l-4 shrink-0", styles.border, styles.bg)}>
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
                            <div className={cn("text-4xl font-bold tracking-tight", styles.text)}>
                                {prediction.risk_level.toUpperCase()}
                            </div>
                            <div className="text-sm text-muted-foreground">
                                Probability of churn: <span className="font-medium text-foreground">{(prediction.probability * 100).toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>

                    <div className="mt-4 pt-4 border-t">
                        <p className="text-sm text-muted-foreground mb-2">
                            Action Recommendations:
                        </p>
                        <ul className="space-y-1">
                            {dynamicRecommendations.map((rec, idx) => (
                                <li key={idx} className="text-sm font-medium text-foreground flex items-start gap-2">
                                    <span className="text-muted-foreground shrink-0">{idx + 1}.</span>
                                    <span>{rec}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                </CardContent>
            </Card>

            {explanation && (
                <Card className="flex-1 flex flex-col">
                    <CardHeader>
                        <CardTitle className="text-xl font-bold">Key Drivers (SHAP)</CardTitle>
                        <CardDescription>
                            Factors contributing most to this prediction.
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="flex-1 pb-4 flex flex-col min-h-0">
                        <div className="flex-1 w-full mt-4">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    layout="vertical"
                                    data={chartData}
                                    margin={{ left: 10, right: 20, top: 0, bottom: 0 }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" horizontal={false} strokeOpacity={0.1} />
                                    <XAxis type="number" hide />
                                    <YAxis
                                        dataKey="name"
                                        type="category"
                                        width={180}
                                        tick={<CustomYAxisTick />}
                                        axisLine={false}
                                        tickLine={false}
                                        interval={0}
                                    />
                                    <Tooltip
                                        cursor={{ fill: 'currentColor', opacity: 0.05 }}
                                        contentStyle={{
                                            borderRadius: '8px',
                                            border: 'none',
                                            boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                                        }}
                                        formatter={(_value: any, _name: any, props: any) => [
                                            `${props.payload.percentage > 0 ? "+" : ""}${props.payload.percentage.toFixed(1)}%`,
                                            "Impact Contribution",
                                        ]}
                                    />
                                    <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={20}>
                                        {chartData.map((entry, index) => (
                                            <Cell
                                                key={`cell-${index}`}
                                                fill={entry.impact === 'increase' ? '#ef4444' : '#22c55e'}
                                                className="transition-all duration-300 hover:opacity-80"
                                            />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="flex justify-center gap-8 text-xs font-medium text-muted-foreground mt-4 pt-4 border-t shrink-0">
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-red-500 rounded-full shadow-sm shadow-red-500/20"></div>
                                <span>Increases Risk</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-green-500 rounded-full shadow-sm shadow-green-500/20"></div>
                                <span>Decreases Risk</span>
                            </div>
                        </div>
                    </CardContent>
                </Card>
            )}
        </div>
    );
}
;
