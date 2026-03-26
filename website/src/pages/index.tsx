import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

const benchmarkStats = [
  {
    value: '1,200',
    label: 'Benchmark prompts',
    note: 'from the current `openedit_bench/metadata.jsonl` snapshot',
  },
  {
    value: '23',
    label: 'Task families',
    note: 'covering object, human, text, chart, hybrid, and reference-based edits',
  },
  {
    value: '17',
    label: 'Editing models',
    note: 'available in the benchmark candidate pool',
  },
];

const taskBands = [
  {
    name: 'Open-set and Hybrid',
    count: '100 + 100',
    detail: 'broad instruction-following edits with composite difficulty',
  },
  {
    name: 'Text and Portrait',
    count: '87 + 62',
    detail: 'in-image text editing and portrait beautification',
  },
  {
    name: 'Camera and Enhancement',
    count: '60 + 60',
    detail: 'viewpoint changes and image-improvement tasks',
  },
];

const modelStrip = [
  'BAGEL',
  'FLUX1_Kontext_dev',
  'FLUX2_dev',
  'GLM_Image',
  'GPT_Image_1p5',
  'Qwen_Image_Edit_2511',
  'Seedream4p5',
  'Step1X_Edit_v1p2',
];

const caseStudies = [
  {
    task: 'Background Change',
    instruction: 'Replace the background with an airport terminal interior.',
    focus: 'Scene replacement while preserving the foreground subject.',
  },
  {
    task: 'Chart Editing',
    instruction: 'Add horizontal yellow dashed lines at the mean of each distribution and include a legend labeled "Mean".',
    focus: 'Structured editing with text and chart semantics.',
  },
  {
    task: 'Motion Change',
    instruction: "Change the boy's pose to an open, communicative gesture with arms spread and hands open, and a speaking expression.",
    focus: 'Human pose and expression transformation with local consistency.',
  },
];

const docsCards = [
  {
    title: 'Pipeline Docs',
    desc: 'Read the runtime overview, workflow entry points, and output artifact structure.',
    href: '/docs/intro',
  },
  {
    title: 'Quickstart',
    desc: 'Go straight to the shortest path for running `annotation`, `eval`, and `train-pairs`.',
    href: '/docs/getting-started/quickstart-autopipeline',
  },
  {
    title: 'Components',
    desc: 'Inspect modules, primitives, and extension surfaces in a source-aligned structure.',
    href: '/docs/components/overview',
  },
];

function HeroSection(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <section className={styles.hero}>
      <div className={styles.heroCopy}>
        <p className={styles.heroKicker}>Paper Landing Page</p>
        <Heading as="h1" className={styles.heroTitle}>
          {siteConfig.title}: paper overview, benchmark, and qualitative results
        </Heading>
        <p className={styles.heroSubtitle}>
          This homepage is reserved for the academic presentation layer. Replace the placeholders below with your final paper title, abstract, benchmark figure, performance chart, and qualitative results.
        </p>
        <p className={styles.heroAbstract}>
          Placeholder abstract: summarize the problem setting, the benchmark contribution, the evaluation or data-construction pipeline, and the main empirical findings here. Keep the final version to one concise research-style paragraph.
        </p>
        <div className={styles.heroActions}>
          <Link className="clean-btn clean-btn--primary" to="/docs/intro">
            Open Pipeline Docs
          </Link>
          <Link className="clean-btn clean-btn--ghost" to="/docs/getting-started/quickstart-autopipeline">
            Open Quickstart
          </Link>
          <Link className="clean-btn clean-btn--ghost" href="https://github.com/ZhangqiJiang07/GEditBench_v2">
            View Repo
          </Link>
        </div>
        <div className={styles.statRibbon}>
          {benchmarkStats.map((item) => (
            <div key={item.label} className={styles.statChip}>
              <strong>{item.value}</strong>
              <span>{item.label}</span>
            </div>
          ))}
        </div>
      </div>
      <div className={styles.heroFigure}>
        <p className={styles.panelLabel}>Paper Figure Placeholder</p>
        <div className={styles.figurePlaceholder}>
          <span>Replace with teaser figure, system diagram, or the main benchmark overview figure.</span>
        </div>
        <p className={styles.figureNote}>
          Current benchmark metadata already exposes 1,200 prompts across 23 task families and 17 editing models, so this area is ready to be wired to your final paper assets later.
        </p>
      </div>
    </section>
  );
}

function BenchmarkSection(): ReactNode {
  return (
    <section className={styles.section}>
      <div className={styles.sectionHeader}>
        <p className={styles.sectionKicker}>Benchmark</p>
        <Heading as="h2" className={styles.sectionTitle}>
          A paper-facing benchmark section with real dataset coverage and placeholder figures
        </Heading>
        <p className={styles.sectionLead}>
          This section is the right place for the benchmark story: scope, task coverage, compared models, and the figure that explains why the benchmark is difficult.
        </p>
      </div>
      <div className={styles.splitGrid}>
        <div className={styles.figureCard}>
          <div className={styles.figurePlaceholder}>
            <span>Benchmark taxonomy figure placeholder</span>
          </div>
          <p className={styles.figureNote}>
            Recommended replacement: a task taxonomy figure or a benchmark composition diagram from the paper.
          </p>
        </div>
        <div className={styles.figureCard}>
          <div className={styles.statsGrid}>
            {benchmarkStats.map((item) => (
              <article key={item.label} className={styles.statCard}>
                <h3>{item.value}</h3>
                <p>{item.label}</p>
                <span>{item.note}</span>
              </article>
            ))}
          </div>
          <div className={styles.taxonomyList}>
            {taskBands.map((band) => (
              <article key={band.name} className={styles.taxonomyCard}>
                <strong>{band.name}</strong>
                <span>{band.count}</span>
                <p>{band.detail}</p>
              </article>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function PerformanceSection(): ReactNode {
  return (
    <section className={styles.section}>
      <div className={styles.sectionHeader}>
        <p className={styles.sectionKicker}>Performance</p>
        <Heading as="h2" className={styles.sectionTitle}>
          Reserve this block for the model-performance figure and benchmark summary table
        </Heading>
        <p className={styles.sectionLead}>
          Replace the chart placeholder with the final leaderboard or benchmark result figure from the paper. The surrounding copy should explain what model is being evaluated, which benchmark split is used, and what metric the audience should focus on.
        </p>
      </div>
      <div className={styles.performanceGrid}>
        <div className={styles.figureCard}>
          <div className={styles.chartPlaceholder}>
            <span>Model performance chart placeholder</span>
          </div>
          <p className={styles.figureNote}>
            Suggested replacement: the main paper chart, such as overall accuracy, win rate, or benchmark score across compared methods.
          </p>
        </div>
        <div className={styles.figureCard}>
          <p className={styles.panelLabel}>Compared Models in the Current Benchmark Snapshot</p>
          <div className={styles.modelList}>
            {modelStrip.map((model) => (
              <span key={model}>{model}</span>
            ))}
          </div>
          <div className={styles.callout}>
            <strong>Optional real data hook</strong>
            <p>
              A local evaluation snapshot already exists in this workspace. When you are ready, this placeholder can be replaced by a real chart derived from the current benchmark outputs instead of a static paper figure.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function CaseSection(): ReactNode {
  return (
    <section className={styles.section}>
      <div className={styles.sectionHeader}>
        <p className={styles.sectionKicker}>Qualitative Cases</p>
        <Heading as="h2" className={styles.sectionTitle}>
          Hold space for the qualitative case studies you will provide later
        </Heading>
        <p className={styles.sectionLead}>
          These cards are intentionally structured so you can swap in real case figures later without changing the page layout.
        </p>
      </div>
      <div className={styles.caseGrid}>
        {caseStudies.map((item) => (
          <article key={item.task} className={styles.caseCard}>
            <div className={styles.caseFigure}>
              <div className={styles.caseSlot}>Source</div>
              <div className={styles.caseSlot}>Candidate A</div>
              <div className={styles.caseSlot}>Candidate B</div>
            </div>
            <p className={styles.caseTask}>{item.task}</p>
            <h3>{item.instruction}</h3>
            <p>{item.focus}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function DocsSection(): ReactNode {
  return (
    <section className={styles.section}>
      <div className={styles.sectionHeader}>
        <p className={styles.sectionKicker}>Docs</p>
        <Heading as="h2" className={styles.sectionTitle}>
          The runtime and source-level documentation now lives under Pipeline Docs
        </Heading>
        <p className={styles.sectionLead}>
          The homepage is for the paper narrative. The actual usage guide, workflow walkthroughs, and source-aligned module pages live under the documentation sidebar.
        </p>
      </div>
      <div className={styles.docsGrid}>
        {docsCards.map((card) => (
          <Link key={card.title} className={styles.docCard} to={card.href}>
            <h3>{card.title}</h3>
            <p>{card.desc}</p>
          </Link>
        ))}
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title} | Paper Overview`}
      description="Paper-oriented landing page for GEditBench v2, with placeholder sections for benchmark figures, performance charts, qualitative cases, and pipeline docs.">
      <main className={styles.page}>
        <HeroSection />
        <BenchmarkSection />
        <PerformanceSection />
        <CaseSection />
        <DocsSection />
        <section className={styles.bottomPanel}>
          <div>
            <p className={styles.sectionKicker}>Next</p>
            <Heading as="h2" className={styles.bottomTitle}>
              Replace the placeholders with paper assets later. For now, the runtime guide starts in Pipeline Docs.
            </Heading>
          </div>
          <div className={styles.bottomActions}>
            <Link className="clean-btn clean-btn--primary" to="/docs/intro">
              Open Pipeline Docs
            </Link>
            <Link className="clean-btn clean-btn--ghost" to="/docs/components/overview">
              Browse Components
            </Link>
          </div>
        </section>
      </main>
    </Layout>
  );
}
