import React from 'react';
import { Header } from './Header.tsx';
import { Footer } from './Footer.tsx';

interface ShellProps {
    children: React.ReactNode;
}

export function Shell({ children }: ShellProps) {
    return (
        <div className="flex min-h-screen flex-col bg-background font-sans antialiased">
            <Header />
            <main className="flex-1 flex flex-col">{children}</main>
            <Footer />
        </div>
    );
}

