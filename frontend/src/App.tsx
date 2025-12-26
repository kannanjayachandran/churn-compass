import { useState } from 'react';
import { Loader2 } from 'lucide-react';
import { Shell } from '@/components/layout/Shell';
import { PredictForm } from '@/components/features/PredictForm';
import { ResultsDisplay } from '@/components/features/ResultsDisplay';
import { UploadForm } from '@/components/features/UploadForm';
import { TopKView } from '@/components/features/TopKView';
import { SystemStatusView } from '@/components/features/SystemStatusView';
import { Button } from '@/components/ui/button';
import { CustomerInput, ExplanationResponse, PredictionResponse } from '@/api/types';
import { api } from '@/api/client';

// Implementing Tabs locally here or in a component file. 
// I'll assume I have Tabs or I'll implement a simple switcher to avoid more files.
// Let's implement simple state switching for now to be safe, or use the Radix Tabs if I had them.
// I will just use a custom Tab implementation to save file count and ensure it works without Radix if user fails to install properly.
// Actually I added @radix-ui/react-tabs to package.json so I should use it.
// I'll create components/ui/tabs.tsx quickly.

const Home = () => {
    const [showSystemStatus, setShowSystemStatus] = useState(false);

    if (showSystemStatus) {
        return (
            <Shell>
                <SystemStatusView onBack={() => setShowSystemStatus(false)} />
            </Shell>
        );
    }

    return (
        <Shell>
            {/* Hero Section */}
            <section className="space-y-6 pb-8 pt-6 md:pb-12 md:pt-10 lg:py-32 bg-slate-50 dark:bg-slate-900 border-b">
                <div className="container flex max-w-[64rem] flex-col items-center gap-4 text-center">
                    <h1 className="font-sans text-4xl font-extrabold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl text-slate-900 dark:text-slate-50">
                        Detect Churn <span className="text-primary"> Risks Early </span>
                    </h1>
                    <p className="max-w-[42rem] leading-normal text-muted-foreground sm:text-xl sm:leading-8">
                        Retail banking revenue leaks when you react too late.
                        Churn Compass transforms ML predictions into explainable, earlier retention actions.
                    </p>
                    <div className="space-x-4">
                        <Button size="lg" onClick={() => document.getElementById('activation')?.scrollIntoView({ behavior: 'smooth' })}>
                            Start Prediction
                        </Button>
                        <Button variant="outline" size="lg" onClick={() => setShowSystemStatus(true)}>
                            System Status
                        </Button>
                    </div>
                </div>
            </section>

            {/* Problem/Agitation Grid */}
            <section className="container space-y-6 bg-slate-50/50 py-8 dark:bg-transparent md:py-12 lg:py-24">
                <div className="mx-auto grid justify-center gap-4 sm:grid-cols-2 md:max-w-[64rem] md:grid-cols-3">
                    <div className="relative overflow-hidden rounded-lg border bg-background p-2">
                        <div className="flex h-[180px] flex-col justify-between rounded-md p-6">
                            <h3 className="font-bold">The Blind Spot</h3>
                            <p className="text-sm text-muted-foreground">Dashboards show you who left yesterday. You need to know who leaves tomorrow.</p>
                        </div>
                    </div>
                    <div className="relative overflow-hidden rounded-lg border bg-background p-2">
                        <div className="flex h-[180px] flex-col justify-between rounded-md p-6">
                            <h3 className="font-bold">Trust Gap</h3>
                            <p className="text-sm text-muted-foreground">Black-box scores are ignored. We provide SHAP-based explanations for every risk score.</p>
                        </div>
                    </div>
                    <div className="relative overflow-hidden rounded-lg border bg-background p-2">
                        <div className="flex h-[180px] flex-col justify-between rounded-md p-6">
                            <h3 className="font-bold">Action Latency</h3>
                            <p className="text-sm text-muted-foreground">Analysis paralysis kills retention. Get instant Top-K lists for immediate outreach.</p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Activation Area */}
            <section id="activation" className="container py-8 md:py-12 lg:py-24">
                <div className="mx-auto flex max-w-[58rem] flex-col items-center justify-center gap-4 text-center mb-10">
                    <h2 className="font-bold text-3xl leading-[1.1] sm:text-3xl md:text-6xl">
                        Intelligence in Action
                    </h2>
                    <p className="max-w-[85%] leading-normal text-muted-foreground sm:text-lg sm:leading-7">
                        Choose your workflow. Predict single customers, analyze bulk CSVs, or scan for top risks.
                    </p>
                </div>

                <div className="mx-auto max-w-5xl">
                    <DashboardTabs />
                </div>
            </section>
        </Shell>
    );
};

// Internal Dashboard Component
const DashboardTabs = () => {
    // Prediction State
    const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
    const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [activeTab, setActiveTab] = useState("single");

    const handleSinglePredict = async (data: CustomerInput) => {
        setIsLoading(true);
        setPrediction(null);
        setExplanation(null);
        try {
            // Parallel calls for speed
            const [pred, expl] = await Promise.all([
                api.predictSingle(data),
                api.explainPrediction(data)
            ]);
            setPrediction(pred);
            setExplanation(expl);
        } catch (e) {
            console.error(e);
            alert("Prediction failed. Ensure backend is running.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="space-y-4">
            {/* Custom Tab Switcher */}
            <div className="flex justify-center p-1 bg-muted rounded-lg w-fit mx-auto">
                {['single', 'batch', 'topk'].map(tab => (
                    <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`px-6 py-2 text-sm font-medium rounded-md transition-all ${activeTab === tab
                            ? 'bg-background text-foreground shadow-sm'
                            : 'text-muted-foreground hover:text-foreground'
                            }`}
                    >
                        {tab === 'single' && "Single Analysis"}
                        {tab === 'batch' && "Batch & CSV"}
                        {tab === 'topk' && "Top Risk Scan"}
                    </button>
                ))}
            </div>

            <div className="mt-8">
                {activeTab === 'single' && (
                    <div className="grid lg:grid-cols-2 gap-8 items-start">
                        <PredictForm onSubmit={handleSinglePredict} isLoading={isLoading} />
                        <div className="space-y-4">
                            {!prediction && !isLoading && (
                                <div className="h-full flex items-center justify-center p-12 border rounded-lg bg-slate-50 border-dashed text-muted-foreground">
                                    Results will appear here
                                </div>
                            )}
                            {isLoading && (
                                <div className="h-full flex items-center justify-center p-12">
                                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                                </div>
                            )}
                            <ResultsDisplay prediction={prediction} explanation={explanation} />
                        </div>
                    </div>
                )}

                {activeTab === 'batch' && (
                    <div className="max-w-xl mx-auto">
                        <UploadForm />
                    </div>
                )}

                {activeTab === 'topk' && (
                    <div className="max-w-3xl mx-auto">
                        <TopKView />
                    </div>
                )}
            </div>
        </div>
    );
};

export default Home;
