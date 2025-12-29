import { useForm, Controller } from "react-hook-form";
import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "@/components/ui/select";
import {
    Card,
    CardContent,
    CardDescription,
    CardFooter,
    CardHeader,
    CardTitle,
} from "@/components/ui/card";
import { CustomerInput } from "@/api/types";

/* ------------------------------------------------------------------ */
/* Schema */
/* ------------------------------------------------------------------ */

const formSchema = z.object({
    CreditScore: z.coerce.number().min(300).max(850),
    Geography: z.enum(["France", "Spain", "Germany"]),
    Gender: z.enum(["Male", "Female"]),
    Age: z.coerce.number().min(10).max(100),
    Tenure: z.coerce.number().min(0).max(10),
    Balance: z.coerce.number().min(0),
    NumOfProducts: z.coerce.number().min(1).max(4),
    HasCrCard: z.coerce.number().min(0).max(1),
    IsActiveMember: z.coerce.number().min(0).max(1),
    EstimatedSalary: z.coerce.number().min(0).max(200000),
    CardType: z.enum(["Silver", "Gold", "Diamond", "Platinum"]),
});

type FormData = z.infer<typeof formSchema>;

interface PredictFormProps {
    onSubmit: (data: CustomerInput) => void;
    isLoading: boolean;
}

export function PredictForm({ onSubmit, isLoading }: PredictFormProps) {
    const {
        register,
        control,
        handleSubmit,
        reset,
        formState: { errors },
    } = useForm<any>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            CreditScore: 650,
            Geography: "Germany",
            Gender: "Female",
            Age: 46,
            Tenure: 4,
            Balance: 100000,
            NumOfProducts: 3,
            HasCrCard: 1,
            IsActiveMember: 0,
            EstimatedSalary: 80000,
            CardType: "Diamond",
        },
    });

    const onFormSubmit = (data: FormData) => {
        onSubmit(data as CustomerInput); // safe: schema matches API contract
    };

    return (
        <Card className="w-full max-w-2xl mx-auto shadow-sm border-border">
            <CardHeader>
                <CardTitle>Single Customer Prediction</CardTitle>
                <CardDescription>
                    Enter customer metrics to generate a real-time risk assessment and
                    explanation.
                </CardDescription>
            </CardHeader>

            <form onSubmit={handleSubmit(onFormSubmit as any)}>
                <CardContent className="grid gap-6 md:grid-cols-2">
                    {/* Credit Score */}
                    <div className="space-y-2">
                        <Label htmlFor="CreditScore">Credit Score</Label>
                        <Input type="number" id="CreditScore" {...register("CreditScore")} />
                        {errors.CreditScore && (
                            <p className="text-xs text-destructive">
                                {String(errors.CreditScore.message)}
                            </p>
                        )}
                    </div>

                    {/* Geography */}
                    <div className="space-y-2">
                        <Label htmlFor="Geography">Geography</Label>
                        <Controller
                            control={control}
                            name="Geography"
                            render={({ field }) => (
                                <Select
                                    onValueChange={field.onChange}
                                    defaultValue={field.value}
                                >
                                    <SelectTrigger>
                                        <SelectValue placeholder="Select Country" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="France">France</SelectItem>
                                        <SelectItem value="Spain">Spain</SelectItem>
                                        <SelectItem value="Germany">Germany</SelectItem>
                                    </SelectContent>
                                </Select>
                            )}
                        />
                        {errors.Geography && (
                            <p className="text-xs text-destructive">
                                {String(errors.Geography.message)}
                            </p>
                        )}
                    </div>

                    {/* Gender */}
                    <div className="space-y-2">
                        <Label htmlFor="Gender">Gender</Label>
                        <Controller
                            control={control}
                            name="Gender"
                            render={({ field }) => (
                                <Select
                                    onValueChange={field.onChange}
                                    defaultValue={field.value}
                                >
                                    <SelectTrigger>
                                        <SelectValue placeholder="Select Gender" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="Male">Male</SelectItem>
                                        <SelectItem value="Female">Female</SelectItem>
                                    </SelectContent>
                                </Select>
                            )}
                        />
                        {errors.Gender && (
                            <p className="text-xs text-destructive">
                                {String(errors.Gender.message)}
                            </p>
                        )}
                    </div>

                    {/* Age */}
                    <div className="space-y-2">
                        <Label htmlFor="Age">Age</Label>
                        <Input type="number" id="Age" {...register("Age")} />
                        {errors.Age && (
                            <p className="text-xs text-destructive">
                                {String(errors.Age.message)}
                            </p>
                        )}
                    </div>

                    {/* Tenure */}
                    <div className="space-y-2">
                        <Label htmlFor="Tenure">Tenure (Years)</Label>
                        <Input type="number" id="Tenure" {...register("Tenure")} />
                        {errors.Tenure && (
                            <p className="text-xs text-destructive">
                                {String(errors.Tenure.message)}
                            </p>
                        )}
                    </div>

                    {/* Balance */}
                    <div className="space-y-2">
                        <Label htmlFor="Balance">Balance</Label>
                        <Input
                            type="number"
                            step="0.01"
                            id="Balance"
                            {...register("Balance")}
                        />
                        {errors.Balance && (
                            <p className="text-xs text-destructive">
                                {String(errors.Balance.message)}
                            </p>
                        )}
                    </div>

                    {/* Num Products */}
                    <div className="space-y-2">
                        <Label htmlFor="NumOfProducts">Number of Products</Label>
                        <Input
                            type="number"
                            id="NumOfProducts"
                            {...register("NumOfProducts")}
                        />
                        {errors.NumOfProducts && (
                            <p className="text-xs text-destructive">
                                {String(errors.NumOfProducts.message)}
                            </p>
                        )}
                    </div>

                    {/* Card Type */}
                    <div className="space-y-2">
                        <Label htmlFor="CardType">Card Type</Label>
                        <Controller
                            control={control}
                            name="CardType"
                            render={({ field }) => (
                                <Select
                                    onValueChange={field.onChange}
                                    defaultValue={field.value}
                                >
                                    <SelectTrigger>
                                        <SelectValue placeholder="Card Type" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="Silver">Silver</SelectItem>
                                        <SelectItem value="Gold">Gold</SelectItem>
                                        <SelectItem value="Platinum">Platinum</SelectItem>
                                        <SelectItem value="Diamond">Diamond</SelectItem>
                                    </SelectContent>
                                </Select>
                            )}
                        />
                        {errors.CardType && (
                            <p className="text-xs text-destructive">
                                {String(errors.CardType.message)}
                            </p>
                        )}
                    </div>

                    {/* Has Credit Card */}
                    <div className="space-y-2">
                        <Label htmlFor="HasCrCard">Has Credit Card</Label>
                        <Controller
                            control={control}
                            name="HasCrCard"
                            render={({ field }) => (
                                <Select
                                    onValueChange={(val) => field.onChange(parseInt(val, 10))}
                                    defaultValue={field.value.toString()}
                                >
                                    <SelectTrigger>
                                        <SelectValue placeholder="Has Credit Card" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="1">Yes</SelectItem>
                                        <SelectItem value="0">No</SelectItem>
                                    </SelectContent>
                                </Select>
                            )}
                        />
                        {errors.HasCrCard && (
                            <p className="text-xs text-destructive">
                                {String(errors.HasCrCard.message)}
                            </p>
                        )}
                    </div>

                    {/* Active Member */}
                    <div className="space-y-2">
                        <Label htmlFor="IsActiveMember">Is Active Member</Label>
                        <Controller
                            control={control}
                            name="IsActiveMember"
                            render={({ field }) => (
                                <Select
                                    onValueChange={(val) => field.onChange(parseInt(val, 10))}
                                    defaultValue={field.value.toString()}
                                >
                                    <SelectTrigger>
                                        <SelectValue placeholder="Is Active" />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="1">Yes</SelectItem>
                                        <SelectItem value="0">No</SelectItem>
                                    </SelectContent>
                                </Select>
                            )}
                        />
                        {errors.IsActiveMember && (
                            <p className="text-xs text-destructive">
                                {String(errors.IsActiveMember.message)}
                            </p>
                        )}
                    </div>

                    {/* Estimated Salary */}
                    <div className="space-y-2">
                        <Label htmlFor="EstimatedSalary">Estimated Salary</Label>
                        <Input
                            type="number"
                            step="1000"
                            id="EstimatedSalary"
                            {...register("EstimatedSalary")}
                        />
                        {errors.EstimatedSalary && (
                            <p className="text-xs text-destructive">
                                {String(errors.EstimatedSalary.message)}
                            </p>
                        )}
                    </div>
                </CardContent>

                <CardFooter className="flex justify-end gap-2 bg-muted/40 p-4 border-t">
                    <Button type="button" variant="ghost" onClick={() => reset()}>
                        Clear
                    </Button>
                    <Button type="submit" disabled={isLoading}>
                        {isLoading && (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        )}
                        Analyze Customer
                    </Button>
                </CardFooter>
            </form>
        </Card>
    );
}
