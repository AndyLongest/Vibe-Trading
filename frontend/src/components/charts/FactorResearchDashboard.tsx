import { useEffect, useMemo, useRef } from "react";
import { Link } from "react-router";
import i18n from "@/i18n";
import { echarts } from "@/lib/echarts";
import { getChartTheme } from "@/lib/chart-theme";
import { useThemeDark } from "@/lib/theme-store";
import type { AlphaBenchResult, AlphaBenchRow } from "@/lib/api";

const BLUE = "#3676df";
const ORANGE = "#ff8a3d";
const GREEN = "#3b9b8f";
const SLATE = "#8f98a6";

function copy(en: string, zh: string): string {
  return i18n.language.toLowerCase().startsWith("zh") ? zh : en;
}

function number(value: unknown, digits = 3): string {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(digits) : "—";
}

function percent(value: unknown, digits = 1): string {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? `${(parsed * 100).toFixed(digits)}%` : "—";
}

function categoryLabel(category: AlphaBenchRow["category"]): string {
  if (category === "alive") return copy("Alive", "有效");
  if (category === "reversed") return copy("Reversed", "反向");
  return copy("Dead", "失效");
}

function categoryTone(category: AlphaBenchRow["category"]): string {
  if (category === "alive") return "bg-[#3b9b8f]/10 text-[#28756c] dark:text-[#72c9be]";
  if (category === "reversed") return "bg-[#ff8a3d]/10 text-[#c65d19] dark:text-[#ffac74]";
  return "bg-slate-500/10 text-slate-600 dark:text-slate-300";
}

function useFactorCharts(
  result: AlphaBenchResult,
  scatterRef: React.RefObject<HTMLDivElement | null>,
  themeRef: React.RefObject<HTMLDivElement | null>,
  rankRef: React.RefObject<HTMLDivElement | null>,
) {
  const dark = useThemeDark();

  useEffect(() => {
    const rows = result.rows || [];
    const nodes = [scatterRef.current, themeRef.current, rankRef.current];
    if (nodes.some((node) => !node)) return;

    const chartTheme = getChartTheme();
    const charts = nodes.map((node) => echarts.init(node!));
    const [scatter, themesChart, ranking] = charts;
    const commonAxis = {
      axisLine: { lineStyle: { color: chartTheme.axisColor } },
      axisLabel: { color: chartTheme.textColor, fontSize: 10 },
      splitLine: { lineStyle: { color: chartTheme.gridColor } },
    };

    const categories: AlphaBenchRow["category"][] = ["alive", "reversed", "dead"];
    const colors = { alive: GREEN, reversed: ORANGE, dead: SLATE };
    scatter.setOption({
      backgroundColor: "transparent",
      grid: { left: 18, right: 24, top: 28, bottom: 16, containLabel: true },
      tooltip: {
        trigger: "item",
        backgroundColor: chartTheme.tooltipBg,
        borderColor: chartTheme.tooltipBorder,
        textStyle: { color: chartTheme.tooltipText },
        formatter: (params: { data: [number, number, string, number, number] }) => {
          const [ic, ir, id, positiveRatio, count] = params.data;
          return `<strong>${id}</strong><br/>IC mean ${number(ic, 4)}<br/>IR ${number(ir, 3)}<br/>Positive ${percent(positiveRatio)}<br/>N ${count}`;
        },
      },
      legend: {
        top: 0,
        right: 4,
        textStyle: { color: chartTheme.textColor, fontSize: 10 },
      },
      xAxis: { ...commonAxis, name: "IC mean", nameTextStyle: { color: chartTheme.textColor } },
      yAxis: { ...commonAxis, name: "IR", nameTextStyle: { color: chartTheme.textColor } },
      series: categories.map((category) => ({
        name: categoryLabel(category),
        type: "scatter",
        symbolSize: 10,
        itemStyle: { color: colors[category], opacity: 0.82 },
        data: rows
          .filter((row) => row.category === category)
          .map((row) => [row.ic_mean, row.ir, row.id, row.ic_positive_ratio, row.ic_count]),
      })),
    });

    const themes = Object.keys(result.by_theme || {}).sort();
    themesChart.setOption({
      backgroundColor: "transparent",
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      legend: { top: 0, right: 4, textStyle: { color: chartTheme.textColor, fontSize: 10 } },
      grid: { left: 12, right: 18, top: 30, bottom: 12, containLabel: true },
      xAxis: {
        ...commonAxis,
        type: "category",
        data: themes.map((theme) => i18n.t(`alphaZoo.themes.${theme}`, { defaultValue: theme })),
        axisLabel: { color: chartTheme.textColor, fontSize: 10, rotate: themes.length > 6 ? 28 : 0 },
      },
      yAxis: { ...commonAxis, type: "value", minInterval: 1 },
      series: categories.map((category) => ({
        name: categoryLabel(category),
        type: "bar",
        stack: "status",
        barMaxWidth: 36,
        itemStyle: { color: colors[category] },
        data: themes.map((theme) => result.by_theme[theme]?.[category] || 0),
      })),
    });

    const topRows = [...rows].sort((a, b) => b.ir - a.ir).slice(0, 12).reverse();
    ranking.setOption({
      backgroundColor: "transparent",
      grid: { left: 12, right: 24, top: 12, bottom: 16, containLabel: true },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      xAxis: { ...commonAxis, type: "value", name: "IR", nameTextStyle: { color: chartTheme.textColor } },
      yAxis: {
        ...commonAxis,
        type: "category",
        data: topRows.map((row) => row.id),
        axisLabel: { color: chartTheme.textColor, fontSize: 10, fontFamily: "monospace" },
      },
      series: [{
        type: "bar",
        barMaxWidth: 18,
        data: topRows.map((row) => ({
          value: row.ir,
          itemStyle: { color: row.ir >= 0 ? BLUE : ORANGE },
        })),
      }],
    });

    let frame: number | null = null;
    const observer = new ResizeObserver(() => {
      if (frame !== null) cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => charts.forEach((chart) => chart.resize()));
    });
    nodes.forEach((node) => observer.observe(node!));
    return () => {
      observer.disconnect();
      if (frame !== null) cancelAnimationFrame(frame);
      charts.forEach((chart) => chart.dispose());
    };
  }, [result, dark, scatterRef, themeRef, rankRef]);
}

function FigureTitle({ index, title, note }: { index: string; title: string; note: string }) {
  return (
    <div className="border-l-4 border-[#3676df] pl-3">
      <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-[#3676df]">Figure {index}</p>
      <h3 className="mt-0.5 text-sm font-semibold text-foreground">{title}</h3>
      <p className="mt-1 text-xs text-muted-foreground">{note}</p>
    </div>
  );
}

export function FactorResearchDashboard({
  result,
  zoo,
  universe,
  period,
  alphaIds,
}: {
  result: AlphaBenchResult;
  zoo: string;
  universe: string;
  period: string;
  alphaIds: string[];
}) {
  const scatterRef = useRef<HTMLDivElement>(null);
  const themeRef = useRef<HTMLDivElement>(null);
  const rankRef = useRef<HTMLDivElement>(null);
  useFactorCharts(result, scatterRef, themeRef, rankRef);

  const rows = useMemo(
    () => [...(result.rows || [])].sort((a, b) => b.ir - a.ir),
    [result.rows],
  );
  const tested = result.n_alphas_tested ?? rows.length;
  const classified = result.alive + result.reversed + result.dead;
  const aliveRatio = classified > 0 ? result.alive / classified : 0;
  const strongest = rows.slice(0, 5);
  const weakest = [...rows].sort((a, b) => a.ic_mean - b.ic_mean).slice(0, 5);
  const metrics = [
    [copy("Tested", "已评测"), tested, BLUE],
    [copy("Alive", "有效"), result.alive, GREEN],
    [copy("Reversed", "反向"), result.reversed, ORANGE],
    [copy("Dead", "失效"), result.dead, SLATE],
    [copy("Skipped", "跳过"), result.skipped ?? 0, "#68707c"],
    [copy("Alive ratio", "有效率"), percent(aliveRatio), BLUE],
  ] as const;

  if (rows.length === 0) {
    return (
      <div className="border border-[#dfe2e7] bg-card p-6 text-sm text-muted-foreground">
        {copy(
          "This result was produced by an older server without per-factor rows. Run the evaluation again to populate the research dashboard.",
          "这份结果由旧版服务生成，没有逐因子明细。请重新运行一次评测以生成研究 Dashboard。",
        )}
      </div>
    );
  }

  return (
    <article className="overflow-hidden border border-[#dfe2e7] bg-card shadow-sm">
      <header className="grid gap-5 border-b border-[#dfe2e7] bg-gradient-to-br from-[#3676df]/[0.07] to-transparent p-5 md:grid-cols-[1fr_auto] md:p-7">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#3676df]">
            {copy("Factor research report", "因子研究报告")}
          </p>
          <h2 className="mt-2 text-xl font-semibold tracking-tight">
            {alphaIds.length ? alphaIds.join(", ") : zoo}
          </h2>
          <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
            {copy(
              "Deterministic IC diagnostics from the completed benchmark. Charts and tables share the same result payload.",
              "来自本次基准评测的确定性 IC 诊断；所有图表和表格共享同一份结果数据。",
            )}
          </p>
        </div>
        <table className="h-fit min-w-64 text-xs">
          <tbody className="[&_td]:border-b [&_td]:border-[#dfe2e7] [&_td]:py-1.5">
            <tr><td className="pr-5 text-muted-foreground">Zoo</td><td className="text-right font-mono">{zoo}</td></tr>
            <tr><td className="pr-5 text-muted-foreground">Universe</td><td className="text-right font-mono">{universe}</td></tr>
            <tr><td className="pr-5 text-muted-foreground">Period</td><td className="text-right font-mono">{period}</td></tr>
          </tbody>
        </table>
      </header>

      <div className="grid grid-cols-2 border-b border-[#dfe2e7] md:grid-cols-3 lg:grid-cols-6">
        {metrics.map(([label, value, color]) => (
          <div key={label} className="border-b border-r border-[#dfe2e7] p-4 last:border-r-0 lg:border-b-0">
            <p className="text-[11px] text-muted-foreground">{label}</p>
            <p className="mt-1 font-mono text-xl font-semibold tabular-nums" style={{ color }}>{value}</p>
          </div>
        ))}
      </div>

      <div className="space-y-8 p-5 md:p-7">
        <section className="space-y-3">
          <FigureTitle index="01" title={copy("IC–IR map", "IC–IR 因子地图")} note={copy("Each point is one factor; colour is the deterministic status classification.", "每个点代表一个因子，颜色表示确定性状态分类。")}/>
          <div ref={scatterRef} className="h-[310px] border border-[#dfe2e7] bg-background/40" role="img" aria-label="IC mean versus IR scatter plot" />
        </section>

        <div className="grid gap-7 lg:grid-cols-2">
          <section className="space-y-3">
            <FigureTitle index="02" title={copy("Status by theme", "主题状态分布")} note={copy("Stacked counts of alive, reversed and dead factors.", "按主题堆叠有效、反向和失效因子数量。")}/>
            <div ref={themeRef} className="h-[300px] border border-[#dfe2e7] bg-background/40" role="img" aria-label="Factor status by theme stacked bar chart" />
          </section>
          <section className="space-y-3">
            <FigureTitle index="03" title={copy("Top information ratios", "最高信息比率")} note={copy("Top twelve factors ranked by IR.", "按 IR 排名前十二的因子。")}/>
            <div ref={rankRef} className="h-[300px] border border-[#dfe2e7] bg-background/40" role="img" aria-label="Top factor information ratios" />
          </section>
        </div>

        <section className="space-y-3">
          <FigureTitle index="04" title={copy("Strongest and weakest evidence", "最强与最弱证据")} note={copy("Separate ranking tables prevent a high IR list from hiding strongly negative IC factors.", "分开列示，避免高 IR 排名掩盖显著负 IC 因子。")}/>
          <div className="grid gap-4 lg:grid-cols-2">
            <CompactFactorTable title={copy("Highest IR", "最高 IR")} rows={strongest} />
            <CompactFactorTable title={copy("Lowest IC mean", "最低 IC 均值")} rows={weakest} />
          </div>
        </section>

        <section className="space-y-3">
          <FigureTitle index="05" title={copy("Complete factor diagnostics", "完整因子诊断")} note={copy("The table is sorted by IR and retains every evaluated factor.", "按 IR 排序并保留每一个已评测因子。")}/>
          <div className="max-h-[520px] overflow-auto border border-[#dfe2e7]">
            <table className="w-full min-w-[900px] text-xs">
              <thead className="sticky top-0 z-10 bg-[#f5f7fa] text-[10px] uppercase tracking-wide text-[#68707c] dark:bg-slate-900">
                <tr className="[&_th]:border-b [&_th]:border-[#dfe2e7] [&_th]:px-3 [&_th]:py-2.5 [&_th]:font-medium">
                  <th className="text-left">Factor</th><th className="text-right">IC mean</th><th className="text-right">IC std</th><th className="text-right">IR</th><th className="text-right">Positive</th><th className="text-right">N</th><th className="text-left">Theme</th><th className="text-left">Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.id} className="border-b border-[#dfe2e7] last:border-0 hover:bg-muted/30 [&_td]:px-3 [&_td]:py-2">
                    <td><Link to={`/alpha-zoo/${encodeURIComponent(row.id)}`} className="font-mono text-[#3676df] hover:underline">{row.id}</Link></td>
                    <td className="text-right font-mono tabular-nums">{number(row.ic_mean, 4)}</td>
                    <td className="text-right font-mono tabular-nums">{number(row.ic_std, 4)}</td>
                    <td className="text-right font-mono font-semibold tabular-nums">{number(row.ir)}</td>
                    <td className="text-right font-mono tabular-nums">{percent(row.ic_positive_ratio)}</td>
                    <td className="text-right font-mono tabular-nums">{row.ic_count}</td>
                    <td className="text-muted-foreground">{(row.theme || []).join(", ") || "—"}</td>
                    <td><span className={`inline-flex px-2 py-0.5 text-[10px] font-medium ${categoryTone(row.category)}`}>{categoryLabel(row.category)}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="space-y-3">
          <FigureTitle index="06" title={copy("Method and boundaries", "方法与边界")} note={copy("What this dashboard proves—and what it does not.", "说明本 Dashboard 能证明和不能证明什么。")}/>
          <div className="grid gap-4 md:grid-cols-3">
            <BoundaryCard title={copy("Signal", "信号")} body={copy("Cross-sectional IC compares factor ranks with forward-return ranks.", "横截面 IC 比较因子排序与未来收益排序。")}/>
            <BoundaryCard title={copy("Classification", "分类")} body={copy("Alive, reversed and dead are assigned by fixed benchmark rules, not by an LLM.", "有效、反向和失效由固定基准规则判定，不由 LLM 判断。")}/>
            <BoundaryCard title={copy("Not a PnL backtest", "并非盈亏回测")} body={copy("No portfolio construction or fills are produced, so equity and drawdown charts are intentionally absent.", "本评测未构造组合或模拟成交，因此刻意不展示净值和回撤图。")}/>
          </div>
          {result.meta && Object.keys(result.meta).length > 0 && (
            <div className="overflow-x-auto border border-[#dfe2e7]">
              <table className="w-full text-xs"><tbody>{Object.entries(result.meta).map(([key, value]) => <tr key={key} className="border-b border-[#dfe2e7] last:border-0"><td className="w-1/3 bg-[#f5f7fa] px-3 py-2 font-mono text-muted-foreground dark:bg-slate-900">{key}</td><td className="px-3 py-2 font-mono">{String(value)}</td></tr>)}</tbody></table>
            </div>
          )}
        </section>
      </div>
    </article>
  );
}

function CompactFactorTable({ title, rows }: { title: string; rows: AlphaBenchRow[] }) {
  return (
    <div className="overflow-hidden border border-[#dfe2e7]">
      <h4 className="border-b border-[#dfe2e7] bg-[#f5f7fa] px-3 py-2 text-xs font-semibold dark:bg-slate-900">{title}</h4>
      <table className="w-full text-xs"><tbody>{rows.map((row) => <tr key={row.id} className="border-b border-[#dfe2e7] last:border-0"><td className="px-3 py-2"><Link to={`/alpha-zoo/${encodeURIComponent(row.id)}`} className="font-mono text-[#3676df] hover:underline">{row.id}</Link></td><td className="px-3 py-2 text-right font-mono">IC {number(row.ic_mean, 4)}</td><td className="px-3 py-2 text-right font-mono font-semibold">IR {number(row.ir)}</td></tr>)}</tbody></table>
    </div>
  );
}

function BoundaryCard({ title, body }: { title: string; body: string }) {
  return <div className="border border-[#dfe2e7] p-4"><h4 className="text-xs font-semibold text-[#2156ae] dark:text-[#7da8ee]">{title}</h4><p className="mt-2 text-xs leading-5 text-muted-foreground">{body}</p></div>;
}
