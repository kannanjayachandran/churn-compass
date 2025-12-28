export function parseCsv(text: string): Record<string, unknown>[] {
    const normalized = text.trim().replace(/\r\n/g, "\n");
    const lines = normalized.split("\n");
    if (lines.length < 2) return [];

    const headers = lines[0].split(",").map(h => h.trim());
    const data: Record<string, unknown>[] = [];

    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(",");
        if (values.length !== headers.length) continue;

        const row: Record<string, unknown> = {};

        headers.forEach((h, index) => {
            const val = values[index].trim();

            const num = Number(val);
            if (val !== "" && Number.isFinite(num)) {
                row[h] = num;
            } else {
                row[h] = val;
            }
        });

        data.push(row);
    }

    return data;
}
