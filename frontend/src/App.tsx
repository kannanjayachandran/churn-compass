import { useState, useRef } from 'react';
import { Loader2, Eye, ShieldCheck, Zap } from 'lucide-react';
import { Shell } from '@/components/layout/Shell.tsx';
import { PredictForm } from '@/components/features/PredictForm';
import { ResultsDisplay } from '@/components/features/ResultsDisplay';
import { UploadForm } from '@/components/features/UploadForm';
import { TopKView } from '@/components/features/TopKView';
import { SystemStatusView } from '@/components/features/SystemStatusView';
import { Button } from '@/components/ui/button';
import { CustomerInput, ExplanationResponse, PredictionResponse } from '@/api/types';
import { api } from '@/api/client';

const Home = () => {
  const [showSystemStatus, setShowSystemStatus] = useState(false);
  const activationRef = useRef<HTMLDivElement>(null);

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
        <div className="container flex max-w-5xl flex-col items-center gap-4 text-center">
          <h1 className="font-sans text-4xl font-extrabold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl text-slate-900 dark:text-slate-50">
            Detect Churn <span className="text-primary"> Risks Early </span>
          </h1>
          <p className="max-w-2xl leading-normal text-muted-foreground sm:text-xl sm:leading-8">
            Retail banking revenue leaks when you react too late.
            Churn Compass transforms ML predictions into explainable, earlier retention actions.
          </p>
          <div className="space-x-4">
            <Button size="lg" onClick={() => activationRef.current?.scrollIntoView({ behavior: "smooth" })}>
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
        <div className="mx-auto grid justify-center gap-6 sm:grid-cols-2 md:max-w-5xl md:grid-cols-3">
          {/* Card 1: The Blind Spot */}
          <div className="group relative overflow-hidden rounded-xl border bg-background p-6 transition-all duration-300 hover:shadow-lg hover:shadow-primary/5 hover:-translate-y-1">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <div className="relative flex flex-col gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-amber-500 to-orange-600 shadow-md shadow-amber-500/20">
                <Eye className="h-6 w-6 text-white" />
              </div>
              <div className="space-y-2">
                <h3 className="font-bold text-lg">The Blind Spot</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Dashboards show you who left <span className="text-amber-600 dark:text-amber-400 font-medium">yesterday</span>. You need to know who leaves <span className="text-primary font-medium">tomorrow</span>.
                </p>
              </div>
            </div>
          </div>

          {/* Card 2: Trust Gap */}
          <div className="group relative overflow-hidden rounded-xl border bg-background p-6 transition-all duration-300 hover:shadow-lg hover:shadow-primary/5 hover:-translate-y-1">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <div className="relative flex flex-col gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500 to-teal-600 shadow-md shadow-emerald-500/20">
                <ShieldCheck className="h-6 w-6 text-white" />
              </div>
              <div className="space-y-2">
                <h3 className="font-bold text-lg">Trust Gap</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Black-box scores are ignored. We provide <span className="text-emerald-600 dark:text-emerald-400 font-medium">SHAP-based explanations</span> for every risk score.
                </p>
              </div>
            </div>
          </div>

          {/* Card 3: Action Latency */}
          <div className="group relative overflow-hidden rounded-xl border bg-background p-6 transition-all duration-300 hover:shadow-lg hover:shadow-primary/5 hover:-translate-y-1">
            <div className="absolute inset-0 bg-gradient-to-br from-primary/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <div className="relative flex flex-col gap-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 shadow-md shadow-violet-500/20">
                <Zap className="h-6 w-6 text-white" />
              </div>
              <div className="space-y-2">
                <h3 className="font-bold text-lg">Action Latency</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  Analysis paralysis kills retention. Get <span className="text-violet-600 dark:text-violet-400 font-medium">instant Top-K lists</span> for immediate outreach.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Activation Area */}
      <section
        ref={activationRef}
        id="activation"
        className='container py-8 md:py-12 lg:py-24'
      >
        <div className="mx-auto flex max-w-232 flex-col items-center justify-center gap-4 text-center mb-10">
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
type Tab = "single" | "batch" | "topk";
const DashboardTabs = () => {
  // Prediction State
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>("single");
  const [error, setError] = useState<string | null>(null);


  const handleSinglePredict = async (data: CustomerInput) => {
    setIsLoading(true);
    setPrediction(null);
    setExplanation(null);
    setError(null);

    try {
      // Parallel calls for speed
      const [pred, expl] = await Promise.all([
        api.predictSingle(data),
        api.explainPrediction(data),
      ]);
      setPrediction(pred);
      setExplanation(expl);
    } catch (e) {
      console.error(e);
      setError("Prediction failed. Ensure backend is running.");
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <div className="space-y-4">
      {/* Custom Tab Switcher */}
      <div className="flex justify-center p-1 bg-muted rounded-lg w-fit mx-auto">
        {(["single", "batch", "topk"] as Tab[]).map(tab => (
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
          <div className="grid lg:grid-cols-2 gap-8 items-stretch">
            <PredictForm onSubmit={handleSinglePredict} isLoading={isLoading} />
            <div className="space-y-4">
              {error && (
                <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
                  {error}
                </div>
              )}

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
