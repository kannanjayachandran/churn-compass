import React, { useEffect, useState } from 'react';
import { Activity, Server, Cpu, Database, GitBranch, Clock, AlertCircle, ArrowLeft } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { api } from '@/api/client';
import { SystemStatusResponse } from '@/api/types';

interface SystemStatusViewProps {
    onBack: () => void;
}

export const SystemStatusView: React.FC<SystemStatusViewProps> = ({ onBack }) => {
    const [status, setStatus] = useState<SystemStatusResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchStatus = async () => {
            try {
                const data = await api.getSystemStatus();
                setStatus(data);
            } catch (err) {
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
                        <div className="text-2xl font-bold truncate text-sm" title={status.system_info.platform}>
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
                        <div className="space-y-4">
                            {Object.entries(status.metrics).length > 0 ? (
                                Object.entries(status.metrics).map(([key, value]) => (
                                    <div key={key} className="flex items-center justify-between border-b pb-2 last:border-0 last:pb-0">
                                        <span className="font-medium text-sm capitalize">{key.replace(/_/g, ' ')}</span>
                                        <span className="font-mono text-sm">{typeof value === 'number' ? value.toFixed(4) : value}</span>
                                    </div>
                                ))
                            ) : (
                                <p className="text-sm text-muted-foreground">No metrics available for this run.</p>
                            )}
                        </div>
                    </CardContent>
                </Card>

                <Card className="col-span-1">
                    <CardHeader>
                        <CardTitle>Model Parameters</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div className="space-y-4">
                            {Object.entries(status.params).length > 0 ? (
                                Object.entries(status.params).map(([key, value]) => (
                                    <div key={key} className="flex items-center justify-between border-b pb-2 last:border-0 last:pb-0">
                                        <span className="font-medium text-sm capitalize">{key.replace(/_/g, ' ')}</span>
                                        <span className="font-mono text-sm">{String(value)}</span>
                                    </div>
                                ))
                            ) : (
                                <p className="text-sm text-muted-foreground">No parameters tracked.</p>
                            )}
                        </div>
                    </CardContent>
                </Card>
            </div>

            <div className="flex items-center justify-center text-xs text-muted-foreground gap-1 mt-8">
                <Clock className="h-3 w-3" />
                Last updated: {new Date(status.timestamp).toLocaleString()}
            </div>
        </div>
    );
};
