export function parseCsv(text: string): Record<string, any>[] {
    const lines = text.trim().split('\n');
    if (lines.length < 2) return [];

    const headers = lines[0].split(',').map(h => h.trim());
    const data = [];

    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        if (values.length !== headers.length) continue;

        const row: Record<string, any> = {};
        headers.forEach((h, index) => {
            let val: string | number = values[index].trim();
            // simple number detect
            if (!isNaN(Number(val)) && val !== "") {
                val = Number(val);
            }
            row[h] = val;
        });
        data.push(row);
    }
    return data;
}
