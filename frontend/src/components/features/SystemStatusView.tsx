import { useEffect, useState } from 'react';
import { Activity, Server, Cpu, Database, GitBranch, Clock, AlertCircle, ArrowLeft } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { api } from '@/api/client';
import { SystemStatusResponse } from '@/api/types';
import { cn } from '@/lib/utils';

interface SystemStatusViewProps {
    onBack: () => void;
}

export function SystemStatusView({ onBack }: SystemStatusViewProps) {
    const [status, setStatus] = useState<SystemStatusResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);
    const [showAllMetrics, setShowAllMetrics] = useState(false);
    const [showAllParams, setShowAllParams] = useState(false);

    const DISPLAY_LIMIT = 6;

    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const data = await api.getSystemStatus();
                setStatus(data);
            } catch (err: unknown) {
                console.error(err);
                setError("Failed to fetch system status");
            } finally {
                setLoading(false);
            }
        };
        fetchStatus();
    }, []);

    if (loading) {
        return (
            <div className="flex h-[50vh] items-center justify-center">
                <div className="flex flex-col items-center gap-2">
                    <Activity className="h-8 w-8 animate-pulse text-primary" />
                    <p className="text-muted-foreground">Loading system status...</p>
                </div>
            </div>
        );
    }

    if (error || !status) {
        return (
            <div className="flex h-[50vh] flex-col items-center justify-center gap-4 text-center">
                <AlertCircle className="h-12 w-12 text-destructive" />
                <h3 className="text-lg font-semibold">System Unreachable</h3>
                <p className="text-muted-foreground">{error}</p>
                <Button onClick={onBack}>Return Home</Button>
            </div>
        );
    }

    return (
        <div className="space-y-6 animate-in fade-in duration-500 container max-w-5xl py-8">
            <div className="flex items-center gap-4">
                <Button variant="ghost" size="icon" onClick={onBack}>
                    <ArrowLeft className="h-4 w-4" />
                </Button>
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">System Status</h2>
                    <p className="text-muted-foreground">Real-time model and infrastructure metrics</p>
                </div>
                <div className={`ml-auto px-3 py-1 rounded-full text-sm font-medium ${status.status === 'healthy'
                    ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                    : 'bg-red-100 text-red-700'
                    }`}>
                    {status.status.toUpperCase()}
                </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">API  Version</CardTitle>
                        <GitBranch className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">v{status.version}</div>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Platform</CardTitle>
                        <Server className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold truncate" title={status.system_info.platform}>
                            {status.system_info.platform}
                        </div>
                        <p className="text-xs text-muted-foreground">Python {status.system_info.system || ''}</p>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">Memory Usage</CardTitle>
                        <Database className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">
                            {(status.system_info.memory_total_gb - status.system_info.memory_available_gb).toFixed(1)} / {status.system_info.memory_total_gb} GB
                        </div>
                        <p className="text-xs text-muted-foreground">
                            {((1 - status.system_info.memory_available_gb / status.system_info.memory_total_gb) * 100).toFixed(0)}% Used
                        </p>
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                        <CardTitle className="text-sm font-medium">CPU Load</CardTitle>
                        <Cpu className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                        <div className="text-2xl font-bold">{status.system_info.cpu_percent}%</div>
                        <p className="text-xs text-muted-foreground">Current utilization</p>
                    </CardContent>
                </Card>
            </div>

            <div className="grid gap-6 md:grid-cols-2">
                <Card className="col-span-1">
                    <CardHeader>
                        <CardTitle>Model Metrics</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className={cn(
                            "space-y-4 pr-1",
                            (showAllMetrics || Object.entries(status.metrics).length > DISPLAY_LIMIT) && "max-h-[300px] overflow-y-auto custom-scrollbar"
                        )}>
                            {Object.entries(status.metrics).length > 0 ? (
                                (showAllMetrics ? Object.entries(status.metrics) : Object.entries(status.metrics).slice(0, DISPLAY_LIMIT))
                                    .map(([key, value]) => (
                                        <div key={key} className="flex items-center justify-between border-b pb-2 last:border-0 last:pb-0">
                                            <span className="font-medium text-sm capitalize text-muted-foreground">{key.replace(/_/g, ' ')}</span>
                                            <span className="font-mono text-sm font-semibold">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                                        </div>
                                    ))
                            ) : (
                                <p className="text-sm text-muted-foreground">No metrics available for this run.</p>
                            )}
                        </div>
                        {Object.entries(status.metrics).length > DISPLAY_LIMIT && (
                            <Button
                                variant="ghost"
                                size="sm"
                                className="w-full mt-4 text-xs h-8 text-primary hover:text-primary hover:bg-primary/5"
                                onClick={() => setShowAllMetrics(!showAllMetrics)}
                            >
                                {showAllMetrics ? "Show Less" : `View More (${Object.entries(status.metrics).length - DISPLAY_LIMIT} additional)`}
                            </Button>
                        )}
                    </CardContent>
                </Card>

                <Card className="col-span-1">
                    <CardHeader>
                        <CardTitle>Model Parameters</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className={cn(
                            "space-y-4 pr-1",
                            (showAllParams || Object.entries(status.params).length > DISPLAY_LIMIT) && "max-h-[300px] overflow-y-auto custom-scrollbar"
                        )}>
                            {Object.entries(status.params).length > 0 ? (
                                (showAllParams ? Object.entries(status.params) : Object.entries(status.params).slice(0, DISPLAY_LIMIT))
                                    .map(([key, value]) => (
                                        <div key={key} className="flex items-center justify-between border-b pb-2 last:border-0 last:pb-0">
                                            <span className="font-medium text-sm capitalize text-muted-foreground">{key.replace(/_/g, ' ')}</span>
                                            <span className="font-mono text-sm font-semibold truncate max-w-[180px]" title={String(value)}>{String(value)}</span>
                                        </div>
                                    ))
                            ) : (
                                <p className="text-sm text-muted-foreground">No parameters tracked.</p>
                            )}
                        </div>
                        {Object.entries(status.params).length > DISPLAY_LIMIT && (
                            <Button
                                variant="ghost"
                                size="sm"
                                className="w-full mt-4 text-xs h-8 text-primary hover:text-primary hover:bg-primary/5"
                                onClick={() => setShowAllParams(!showAllParams)}
                            >
                                {showAllParams ? "Show Less" : `View More (${Object.entries(status.params).length - DISPLAY_LIMIT} additional)`}
                            </Button>
                        )}
                    </CardContent>
                </Card>
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>Technology Stack</CardTitle>
                </CardHeader>
                <CardContent>
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                        <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-blue-100 dark:bg-blue-900/30">
                                <Database className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                            </div>
                            <div>
                                <p className="font-medium text-sm">MLflow</p>
                                <p className="text-xs text-muted-foreground">Experiment Tracking</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-purple-100 dark:bg-purple-900/30">
                                <Activity className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                            </div>
                            <div>
                                <p className="font-medium text-sm">Optuna</p>
                                <p className="text-xs text-muted-foreground">Hyperparameter Tuning</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-green-100 dark:bg-green-900/30">
                                <GitBranch className="h-5 w-5 text-green-600 dark:text-green-400" />
                            </div>
                            <div>
                                <p className="font-medium text-sm">XGBoost</p>
                                <p className="text-xs text-muted-foreground">Gradient Boosting</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-3 p-3 rounded-lg bg-muted/50">
                            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-orange-100 dark:bg-orange-900/30">
                                <Server className="h-5 w-5 text-orange-600 dark:text-orange-400" />
                            </div>
                            <div>
                                <p className="font-medium text-sm">FastAPI</p>
                                <p className="text-xs text-muted-foreground">REST API Server</p>
                            </div>
                        </div>
                    </div>
                </CardContent>
            </Card>

            <div className="flex items-center justify-center text-xs text-muted-foreground gap-1 mt-8">
                <Clock className="h-3 w-3" />
                Last updated: {new Date(status.timestamp).toLocaleString()}
            </div>
        </div>
    );
};
