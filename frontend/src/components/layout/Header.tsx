import React from 'react';
import { Compass } from 'lucide-react';

export const Header: React.FC = () => {
    return (
        <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container flex h-14 max-w-screen-2xl items-center">
                <div className="flex items-center gap-2 font-bold text-xl tracking-tight text-primary">
                    <Compass className="h-6 w-6 text-primary" />
                    <span>Churn Compass</span>
                </div>
                <div className="ml-auto flex items-center gap-4">
                    <span className="text-sm font-medium text-muted-foreground hidden sm:inline-block">
                        Production Environment
                    </span>
                    <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse" title="System Operational"></div>
                </div>
            </div>
        </header>
    );
};
