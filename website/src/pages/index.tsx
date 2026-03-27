import {useRef, useState, type ReactNode} from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './index.module.css';

type PaperAuthor = {
  name: string;
  affiliations: number[];
};

type PaperAffiliation = {
  id: number;
  name: string;
};

type LeaderboardEntry = {
  rank: number;
  source: 'closed' | 'open';
  model: string;
  samples: string;
  instruction: string;
  quality: string;
  consistency: string;
  overall: string;
  arena: string;
};

type QualitativeCase = {
  title: string;
  image1200: string;
  alt: string;
};

const paperAffiliations: PaperAffiliation[] = [
  {id: 1, name: 'Nanyang Technological University'},
  {id: 2, name: 'StepFun'},
  {id: 3, name: 'Southeast University'},
];

// Fill each author's affiliation ids using the numbered institutions above.
const paperAuthors: PaperAuthor[] = [
  {name: 'Zhangqi Jiang', affiliations: [1, 2]},
  {name: 'Zheng Sun', affiliations: [2]},
  {name: 'Xianfang Zeng', affiliations: [2]},
  {name: 'Yufeng Yang', affiliations: [2]},
  {name: 'Xuanyang Zhang', affiliations: [2]},
  {name: 'Yongliang Wu', affiliations: [3]},
  {name: 'Wei Cheng', affiliations: [2]},
  {name: 'Gang Yu', affiliations: [2]},
  {name: 'Xu Yang', affiliations: [3]},
  {name: 'Bihan Wen', affiliations: [1]},
];

const leaderboardEntries: LeaderboardEntry[] = [
  {
    rank: 1,
    source: 'closed',
    model: 'Nano Banana Pro (26-03-04)',
    samples: '1,156',
    instruction: '1,126 (-13/+15)',
    quality: '1,066 (-9/+10)',
    consistency: '1,108 (-11/+11)',
    overall: '1,096 (-6/+6)',
    arena: '1,251 / #2',
  },
  {
    rank: 2,
    source: 'closed',
    model: 'Seedream 4.5 (26-03-11)',
    samples: '1,190',
    instruction: '1,111 (-12/+12)',
    quality: '1,142 (-11/+11)',
    consistency: '1,030 (-11/+12)',
    overall: '1,089 (-7/+7)',
    arena: '1,196 / #3',
  },
  {
    rank: 3,
    source: 'closed',
    model: 'GPT Image 1.5 (26-03-04)',
    samples: '1,081',
    instruction: '1,260 (-13/+15)',
    quality: '1,149 (-12/+12)',
    consistency: '846 (-13/+13)',
    overall: '1,071 (-7/+6)',
    arena: '1,270 / #1',
  },
  {
    rank: 4,
    source: 'open',
    model: 'FLUX.2 [klein] 9B',
    samples: '1,200',
    instruction: '1,083 (-13/+12)',
    quality: '1,025 (-11/+10)',
    consistency: '1,019 (-10/+9)',
    overall: '1,039 (-6/+6)',
    arena: '1,166 / #4',
  },
  {
    rank: 5,
    source: 'open',
    model: 'Qwen-Image-Edit-2511',
    samples: '1,200',
    instruction: '1,095 (-10/+10)',
    quality: '1,060 (-11/+11)',
    consistency: '972 (-9/+10)',
    overall: '1,038 (-6/+6)',
    arena: '1,164 / #5',
  },
  {
    rank: 6,
    source: 'open',
    model: 'FLUX.2 [klein] 4B',
    samples: '1,200',
    instruction: '1,007 (-12/+12)',
    quality: '1,019 (-10/+10)',
    consistency: '1,070 (-10/+10)',
    overall: '1,031 (-6/+6)',
    arena: '1,107 / #10',
  },
  {
    rank: 7,
    source: 'open',
    model: 'FLUX.2 [dev] Turbo',
    samples: '1,200',
    instruction: '1,068 (-12/+12)',
    quality: '936 (-10/+10)',
    consistency: '1,064 (-11/+10)',
    overall: '1,021 (-6/+6)',
    arena: '1,153 / #6',
  },
  {
    rank: 8,
    source: 'open',
    model: 'Qwen-Image-Edit-2509',
    samples: '1,200',
    instruction: '1,033 (-10/+11)',
    quality: '1,062 (-10/+12)',
    consistency: '955 (-9/+9)',
    overall: '1,014 (-5/+6)',
    arena: '1,142 / #7',
  },
  {
    rank: 9,
    source: 'open',
    model: 'Qwen-Image-Edit',
    samples: '1,200',
    instruction: '991 (-10/+10)',
    quality: '1,073 (-11/+12)',
    consistency: '971 (-11/+11)',
    overall: '1,010 (-6/+6)',
    arena: '1,088 / #12',
  },
  {
    rank: 10,
    source: 'open',
    model: 'FLUX.2 [dev]',
    samples: '1,200',
    instruction: '1,037 (-12/+13)',
    quality: '965 (-10/+10)',
    consistency: '1,018 (-11/+11)',
    overall: '1,006 (-7/+7)',
    arena: '1,137 / #8',
  },
  {
    rank: 11,
    source: 'open',
    model: 'LongCat-Image-Edit',
    samples: '1,200',
    instruction: '1,018 (-10/+11)',
    quality: '968 (-10/+9)',
    consistency: '1,017 (-10/+9)',
    overall: '1,001 (-6/+5)',
    arena: '1,111 / #9',
  },
  {
    rank: 12,
    source: 'open',
    model: 'Step1X-Edit-v1p2',
    samples: '1,200',
    instruction: '909 (-12/+12)',
    quality: '1,007 (-12/+11)',
    consistency: '1,067 (-11/+11)',
    overall: '996 (-6/+7)',
    arena: '1,093 / #11',
  },
  {
    rank: 13,
    source: 'open',
    model: 'GLM-Image',
    samples: '1,200',
    instruction: '787 (-13/+14)',
    quality: '1,023 (-11/+11)',
    consistency: '1,109 (-13/+14)',
    overall: '979 (-6/+6)',
    arena: '930 / #14',
  },
  {
    rank: 14,
    source: 'open',
    model: 'OmniGen V2',
    samples: '1,200',
    instruction: '807 (-13/+12)',
    quality: '910 (-12/+12)',
    consistency: '929 (-13/+13)',
    overall: '888 (-7/+7)',
    arena: '919 / #15',
  },
  {
    rank: 15,
    source: 'open',
    model: 'FLUX.1 Kontext [dev]',
    samples: '1,200',
    instruction: '849 (-13/+13)',
    quality: '900 (-13/+14)',
    consistency: '840 (-14/+13)',
    overall: '869 (-7/+8)',
    arena: '1,017 / #13',
  },
  {
    rank: 16,
    source: 'open',
    model: 'Bagel',
    samples: '1,200',
    instruction: '820 (-13/+13)',
    quality: '694 (-17/+16)',
    consistency: '987 (-13/+14)',
    overall: '851 (-8/+8)',
    arena: '915 / #16',
  },
];

const heroAbstract: ReactNode = (
  <>
    Recent advances in image editing have enabled models to handle complex
    instructions with impressive realism. However, existing evaluation
    frameworks lag behind: current benchmarks suffer from narrow task coverage,
    while standard metrics fail to adequately capture visual consistency, i.e.,
    the preservation of identity, structure and semantic coherence between
    edited and original images. To address these limitations, we introduce
    <strong> GEditBench v2</strong>, a comprehensive benchmark with 1,200
    real-world user queries spanning 23 tasks, including a dedicated open-set
    category for unconstrained, out-of-distribution editing instructions beyond
    predefined tasks. Furthermore, we propose <strong>PVC-Judge</strong>, an
    open-source pairwise assessment model for visual consistency, trained via
    two novel region-decoupled preference data synthesis pipelines. Besides, we
    construct <strong>VCReward-Bench</strong> using expert-annotated preference
    pairs to assess the alignment of PVC-Judge with human judgments on visual
    consistency evaluation. Experiments show that our PVC-Judge achieves
    state-of-the-art evaluation performance among open-source models and even
    surpasses <em>GPT-5.1</em> on average. Finally, by benchmarking 16 frontier
    editing models, we show that GEditBench v2 enables more human-aligned
    evaluation, revealing critical limitations of current models, and providing
    a reliable foundation for advancing precise image editing.
  </>
);

const heroLinks = [
  {
    label: 'arXiv',
    href: 'https://arxiv.org/',
  },
  {
    label: 'GEditBench v2',
    href: 'https://huggingface.co/',
  },
  {
    label: 'VCReward-Bench',
    href: 'https://gitlab.basemind.com/i-jiangzhangqi/open_edit',
  },
  {
    label: 'PVC-Judge',
    href: '/docs/intro',
  },
  {
    label: 'Code',
    href: '/docs/intro',
  },
];

const heroFacts = [
  {
    value: '1',
    label: 'GEditBench v2',
    detail: 'comprising 22 predefined tasks with an open-set category to evaluate editing models in real-world scenarios.',
  },
  {
    value: '2',
    label: 'VCReward-Bench',
    detail: 'evaluate assessment models for instruction-guided image editing in visual consistency, supported by 3,506 expert-annotated preference pairs',
  },
  {
    value: '3',
    label: 'PVC-Judge',
    detail: 'a pairwise assessment model for visual consistency',
  },
  {
    value: '4',
    label: 'AutoPipeline',
    detail: 'two novel region-decoupled preference data synthesis pipelines with ',
  },
];

const contributionRows = [
  {
    index: '01',
    title: 'One framework for three operational goals',
    description:
      'AutoPipeline keeps candidate scoring, benchmark judging, and preference-data construction inside a single system instead of scattering them across disconnected scripts.',
  },
  {
    index: '02',
    title: 'Configuration-driven execution',
    description:
      'Most routine usage is expressed through task configs, pipeline configs, and user configs, which reduces the need for repeated runtime code edits.',
  },
  {
    index: '03',
    title: 'Artifacts as the downstream interface',
    description:
      'Grouped JSONL, eval JSONL, and train-pair JSON are treated as first-class outputs so evaluation and data construction stay reproducible and inspectable.',
  },
];

const workflowRows = [
  {
    name: 'annotation',
    title: 'Score candidate edited images with structured metrics',
    description:
      'Use this workflow when you want to compare multiple candidate edits for the same source image and keep the strongest outputs.',
    command: 'autopipeline annotation',
    artifact: '<save-path>/<task>_grouped.jsonl',
  },
  {
    name: 'eval',
    title: 'Run pairwise judging on a benchmark',
    description:
      'Use this workflow when the main question is comparative: which candidate wins, which model performs better, and how the results distribute across a benchmark.',
    command: 'autopipeline eval',
    artifact: '<save-path>/<bmk>/<config>/<timestamp>.jsonl',
  },
  {
    name: 'train-pairs',
    title: 'Convert outputs into preference-learning data',
    description:
      'Use this workflow when grouped results or judge outputs need to be turned into training pairs for downstream alignment and preference-learning pipelines.',
    command: 'autopipeline train-pairs',
    artifact: '<output-dir>/<task>.json',
  },
];

const runtimeFlow = [
  {
    step: '01',
    title: 'CLI',
    description: 'Normalize task, benchmark, and path inputs into one coherent command surface.',
  },
  {
    step: '02',
    title: 'ConfigEngine',
    description: 'Resolve bases, variable references, and merged defaults before execution begins.',
  },
  {
    step: '03',
    title: 'Runner',
    description: 'Load datasets, coordinate execution, and manage cache-aware runtime behavior.',
  },
  {
    step: '04',
    title: 'Pipelines',
    description: 'Dispatch into the correct workflow family and the required modules or judges.',
  },
  {
    step: '05',
    title: 'Artifacts',
    description: 'Emit grouped results, eval results, or train-pair outputs for downstream use.',
  },
];

const pipelineFamilies = [
  {
    name: 'object-centric',
    description:
      'For object addition, removal, replacement, color change, material change, size control, and text-editing style tasks.',
  },
  {
    name: 'human-centric',
    description:
      'For identity consistency, face and body appearance, local human edits, and pose-sensitive evaluation scenarios.',
  },
  {
    name: 'vlm-as-a-judge',
    description:
      'For pairwise comparison, benchmark ranking, and direct preference judgments between candidate outputs.',
  },
];

const resourceRows = [
  {
    title: 'Overview',
    description:
      'Read the documentation entry point for the runtime model, workflow map, and artifact-oriented mental model.',
    href: '/docs/intro',
  },
  {
    title: 'Quickstart',
    description:
      'Jump straight to the shortest successful command path for annotation, eval, and train-pairs.',
    href: '/docs/getting-started/quickstart-autopipeline',
  },
  {
    title: 'Components',
    description:
      'Inspect modules, primitives, and extension surfaces in a source-aligned structure once the high-level picture is clear.',
    href: '/docs/components/overview',
  },
  {
    title: 'Extending',
    description:
      'Learn how to add new pipeline families, modules, and primitives without reverse-engineering the whole codebase.',
    href: '/docs/extending/overview',
  },
];

const qualitativeCases: QualitativeCase[] = [
  {
    title: 'Open-Set Edits',
    image1200: '/paper_img/cases1-1200.webp',
    alt: 'Qualitative analysis case on open-set edits',
  },
  {
    title: 'Weak Perception of Inter-Object Relations.',
    image1200: '/paper_img/cases2-1200.webp',
    alt: 'Qualitative analysis case on weak perception of inter-object relations',
  },
  {
    title: 'Struggle with Small Faces.',
    image1200: '/paper_img/cases3-1200.webp',
    alt: 'Qualitative analysis case on struggle with small faces',
  },
];

const pipelineDocLinks = [
  {
    label: 'Runtime Overview',
    href: '/docs/intro',
  },
  {
    label: 'Framework Overview',
    href: '/docs/tutorial/framework-overview',
  },
  {
    label: 'Pipeline Guide',
    href: '/docs/tutorial/guide-for-pipelines/overview',
  },
  {
    label: 'Components',
    href: '/docs/components/overview',
  },
];

type SectionProps = {
  eyebrow: string;
  title: string;
  lead?: string;
  toneClassName: string;
  wideHeader?: boolean;
  children: ReactNode;
};

function Section({
  eyebrow,
  title,
  lead,
  toneClassName,
  wideHeader = true,
  children,
}: SectionProps): ReactNode {
  return (
    <section className={`${styles.band} ${toneClassName}`}>
      <div className={styles.inner}>
        <div
          className={
            wideHeader
              ? `${styles.bandHeader} ${styles.bandHeaderWide}`
              : styles.bandHeader
          }>
          <p className={styles.eyebrow}>{eyebrow}</p>
          <Heading as="h2" className={styles.bandTitle}>
            {title}
          </Heading>
          {lead ? <p className={styles.bandLead}>{lead}</p> : null}
        </div>
        {children}
      </div>
    </section>
  );
}

function renderAffiliationSuperscript(affiliations: number[]): ReactNode {
  if (affiliations.length === 0) {
    return null;
  }

  return <sup className={styles.affiliationSup}>{affiliations.join(',')}</sup>;
}

function getAuthorPrefix(index: number, total: number): string {
  if (index === 0) {
    return '';
  }

  if (index === total - 1) {
    return total === 2 ? ' and ' : ', and ';
  }

  return ', ';
}

function splitMetric(metric: string): {value: string; ci: string} {
  const match = metric.match(/^(.*?)\s*\((.*?)\)$/);

  if (!match) {
    return {value: metric, ci: ''};
  }

  return {
    value: match[1].trim(),
    ci: match[2].trim(),
  };
}

function splitArena(arena: string): {elo: string; rank: string} {
  const [elo = arena, rank = ''] = arena.split(' / ');
  return {elo, rank};
}

function HeroSection(): ReactNode {
  return (
    <section className={`${styles.band} ${styles.toneA} ${styles.heroBand}`}>
      <div className={styles.inner}>
        <div className={styles.heroHeader}>
          <p className={styles.eyebrow}>Project Page</p>
          <Heading as="h1" className={styles.heroTitle}>
            <>
              GEditBench v2:
              <span className={styles.heroTitleSecondary}>
                A Human-Aligned Benchmark for General Image Editing
              </span>
            </>
          </Heading>
          <p className={styles.authorLine}>
            {paperAuthors.map((author, index) => (
              <span key={author.name}>
                {getAuthorPrefix(index, paperAuthors.length)}
                {author.name}
                {renderAffiliationSuperscript(author.affiliations)}
              </span>
            ))}
          </p>
          <p className={styles.affiliationLine}>
            {paperAffiliations.map((affiliation, index) => (
              <span key={affiliation.id}>
                {index > 0 ? ' | ' : ''}
                {renderAffiliationSuperscript([affiliation.id])} {affiliation.name}
              </span>
            ))}
          </p>
          <div className={styles.heroLinks}>
            {heroLinks.map((item) => (
              <Link key={item.label} className={styles.heroLink} href={item.href}>
                {item.label}
              </Link>
            ))}
          </div>
          <p className={styles.heroAbstract}>{heroAbstract}</p>
        </div>

        <div className={styles.factRow}>
          {heroFacts.map((item) => (
            <article key={item.label} className={styles.factItem}>
              <strong>{item.value}</strong>
              <div>
                <span>{item.label}</span>
                <p>{item.detail}</p>
              </div>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}

function OverviewSection(): ReactNode {
  const teaserImageUrl1200 = useBaseUrl('/paper_img/teaser-1200.webp');
  const teaserImageUrl1800 = useBaseUrl('/paper_img/teaser-1800.webp');

  return (
    <Section
      eyebrow="Teaser"
      title="Overview of GEditBench v2"
      wideHeader
      toneClassName={styles.toneB}>
      <div className={styles.overviewFigure}>
        <img
          className={styles.overviewImage}
          src={teaserImageUrl1800}
          srcSet={`${teaserImageUrl1200} 1200w, ${teaserImageUrl1800} 1800w`}
          sizes="(max-width: 996px) calc(100vw - 1.8rem), 1180px"
          alt="Overview of GEditBench v2 benchmark teaser"
          width={4408}
          height={2475}
          decoding="async"
        />
      </div>
    </Section>
  );
}

function LeaderboardSection(): ReactNode {
  return (
    <Section
      eyebrow="Leaderboard"
      title="Leaderboard on GEditBench v2"
      lead="Models are ranked by overall Elo from pairwise comparisons. Instruction Following and Visual Quality are assessed by GPT-4o, while Visual Consistency is evaluated by PVC-Judge. Confidence intervals come from the leaderboard reported in static/tables/main_res.tex."
      wideHeader
      toneClassName={styles.toneA}>
      <div className={styles.leaderboardIntro}>
        <p className={styles.leaderboardNote}>
          Overall Elo scores and 95% confidence intervals are computed via 1,000
          bootstrap iterations. Arena values below are reported alongside their
          recorded rank on March 26, 2026.
        </p>
      </div>
          <div className={styles.leaderboardTableWrap}>
        <table className={styles.leaderboardTable}>
          <colgroup>
            <col className={styles.rankCol} />
            <col className={styles.modelCol} />
            <col className={styles.sourceCol} />
            <col className={styles.samplesCol} />
            <col className={styles.metricCol} />
            <col className={styles.metricCol} />
            <col className={styles.metricCol} />
            <col className={styles.overallCol} />
            <col className={styles.arenaEloCol} />
            <col className={styles.arenaRankCol} />
          </colgroup>
          <thead>
            <tr>
              <th>Rank</th>
              <th>Model</th>
              <th>Source</th>
              <th>Samples</th>
              <th>
                <span className={styles.headerStack}>
                  <span>Instruction</span>
                  <span>Following</span>
                </span>
              </th>
              <th>
                <span className={styles.headerStack}>
                  <span>Visual</span>
                  <span>Quality</span>
                </span>
              </th>
              <th>
                <span className={styles.headerStack}>
                  <span>Visual</span>
                  <span>Consistency</span>
                </span>
              </th>
              <th>Overall</th>
              <th>
                <span className={styles.headerStack}>
                  <span>Arena</span>
                  <span>Elo</span>
                </span>
              </th>
              <th>
                <span className={styles.headerStack}>
                  <span>Arena</span>
                  <span>Rank</span>
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {leaderboardEntries.map((entry) => {
              const instructionMetric = splitMetric(entry.instruction);
              const qualityMetric = splitMetric(entry.quality);
              const consistencyMetric = splitMetric(entry.consistency);
              const overallMetric = splitMetric(entry.overall);
              const arenaMetric = splitArena(entry.arena);

              return (
                <tr
                  key={entry.model}
                  className={entry.rank <= 3 ? styles.leaderboardTopRow : undefined}>
                  <td className={styles.leaderboardRank}>{entry.rank}</td>
                  <td className={styles.leaderboardModel}>{entry.model}</td>
                  <td>
                    <span
                      className={
                        entry.source === 'closed'
                          ? `${styles.sourceBadge} ${styles.sourceClosed}`
                          : `${styles.sourceBadge} ${styles.sourceOpen}`
                      }>
                      {entry.source === 'closed' ? 'Closed' : 'Open'}
                    </span>
                  </td>
                  <td>{entry.samples}</td>
                  <td className={styles.metricCell}>
                    <span className={styles.metricMain}>{instructionMetric.value}</span>
                    <span className={styles.metricCi}>{instructionMetric.ci}</span>
                  </td>
                  <td className={styles.metricCell}>
                    <span className={styles.metricMain}>{qualityMetric.value}</span>
                    <span className={styles.metricCi}>{qualityMetric.ci}</span>
                  </td>
                  <td className={styles.metricCell}>
                    <span className={styles.metricMain}>{consistencyMetric.value}</span>
                    <span className={styles.metricCi}>{consistencyMetric.ci}</span>
                  </td>
                  <td className={`${styles.metricCell} ${styles.leaderboardOverall}`}>
                    <span className={styles.metricMain}>{overallMetric.value}</span>
                    <span className={styles.metricCi}>{overallMetric.ci}</span>
                  </td>
                  <td>{arenaMetric.elo}</td>
                  <td>{arenaMetric.rank}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function AutoPipelineSection(): ReactNode {
  const pipelineImageUrl1200 = useBaseUrl('/paper_img/anno_pipelines-1200.webp');
  const pipelineImageUrl1800 = useBaseUrl('/paper_img/anno_pipelines-1800.webp');

  return (
    <Section
      eyebrow="AutoPipeline"
      title="AutoPipeline for structured image-edit evaluation"
      lead="AutoPipeline organizes evaluation into task-adaptive pipelines. For object-centric edits, it spatially decouples edited and non-edited regions before applying region-specific metrics; for human-centric edits, it routes face, body, and hair related changes into dedicated evaluation branches and rubric-aware outputs."
      wideHeader
      toneClassName={styles.toneB}>
      <div className={styles.pipelineFigure}>
        <img
          className={styles.pipelineImage}
          src={pipelineImageUrl1800}
          srcSet={`${pipelineImageUrl1200} 1200w, ${pipelineImageUrl1800} 1800w`}
          sizes="(max-width: 996px) calc(100vw - 1.8rem), 1180px"
          alt="AutoPipeline object-centric and human-centric pipeline overview"
          width={4458}
          height={2897}
          loading="lazy"
          decoding="async"
        />
      </div>
      <div className={styles.pipelineCtaRow}>
        {pipelineDocLinks.map((item) => (
          <Link key={item.label} className={styles.pipelineCta} to={item.href}>
            {item.label}
          </Link>
        ))}
      </div>
    </Section>
  );
}

function PVCJudgeSection(): ReactNode {
  const rewardImageUrl1200 = useBaseUrl('/paper_img/reward_performance-1200.webp');
  const rewardImageUrl1800 = useBaseUrl('/paper_img/reward_performance-1800.webp');

  return (
    <Section
      eyebrow="PVC-Judge"
      title="PVC-Judge performance on EditScore Reward-Bench and VCReward-Bench"
      lead="We evaluate PVC-Judge on two complementary reward benchmarks. The radar plots compare pairwise judges and pointwise scoring baselines across diverse editing tasks, showing that PVC-Judge delivers consistently strong alignment with human preference, especially on visual consistency sensitive categories."
      toneClassName={styles.toneA}>
      <div className={styles.rewardFigure}>
        <img
          className={styles.rewardImage}
          src={rewardImageUrl1800}
          srcSet={`${rewardImageUrl1200} 1200w, ${rewardImageUrl1800} 1800w`}
          sizes="(max-width: 996px) calc(100vw - 1.8rem), 1180px"
          alt="PVC-Judge performance on EditScore Reward-Bench and VCReward-Bench"
          width={3789}
          height={2086}
          loading="lazy"
          decoding="async"
        />
      </div>
    </Section>
  );
}

function QualitativeSection(): ReactNode {
  const carouselRef = useRef<HTMLDivElement | null>(null);
  const slideRefs = useRef<Array<HTMLElement | null>>([]);
  const [currentCase, setCurrentCase] = useState(0);

  const scrollToCase = (targetIndex: number) => {
    const node = carouselRef.current;
    const targetSlide = slideRefs.current[targetIndex];

    if (!node || !targetSlide) {
      return;
    }

    const centeredLeft =
      targetSlide.offsetLeft - (node.clientWidth - targetSlide.clientWidth) / 2;

    node.scrollTo({left: centeredLeft, behavior: 'smooth'});
    setCurrentCase(targetIndex);
  };

  const scrollCarousel = (direction: 'prev' | 'next') => {
    const nextIndex =
      direction === 'next'
        ? Math.min(currentCase + 1, qualitativeCases.length - 1)
        : Math.max(currentCase - 1, 0);

    scrollToCase(nextIndex);
  };

  const handleCarouselScroll = () => {
    const node = carouselRef.current;

    if (!node) {
      return;
    }

    const viewportCenter = node.scrollLeft + node.clientWidth / 2;
    let nextIndex = currentCase;
    let smallestDistance = Number.POSITIVE_INFINITY;

    slideRefs.current.forEach((slide, index) => {
      if (!slide) {
        return;
      }

      const slideCenter = slide.offsetLeft + slide.clientWidth / 2;
      const distance = Math.abs(slideCenter - viewportCenter);

      if (distance < smallestDistance) {
        smallestDistance = distance;
        nextIndex = index;
      }
    });

    if (nextIndex !== currentCase) {
      setCurrentCase(nextIndex);
    }
  };

  return (
    <Section
      eyebrow="Qualitative Analysis"
      title="Representative failure cases from GEditBench v2"
      lead="We highlight several recurring failure modes revealed by human-aligned evaluation, including open-set reasoning errors, weak perception of inter-object relations, and difficulty handling fine facial details at small scale."
      toneClassName={styles.toneB}>
      <div className={styles.qualitativeStage}>
        <button
          type="button"
          className={`${styles.carouselButton} ${styles.carouselButtonLeft}`}
          onClick={() => scrollCarousel('prev')}>
          disabled={currentCase === 0}
          Prev
        </button>
        <div
          ref={carouselRef}
          className={styles.qualitativeCarousel}
          onScroll={handleCarouselScroll}>
          <div className={styles.qualitativeSpacer} aria-hidden="true" />
          {qualitativeCases.map((item, index) => {
          const imageUrl = useBaseUrl(item.image1200);

          return (
            <article
              key={item.title}
              ref={(node) => {
                slideRefs.current[index] = node;
              }}
              className={
                index === currentCase
                  ? `${styles.qualitativeSlide} ${styles.qualitativeSlideActive}`
                  : styles.qualitativeSlide
              }>
              <p className={styles.qualitativeLabel}>{item.title}</p>
              <img
                className={styles.qualitativeImage}
                src={imageUrl}
                alt={item.alt}
                loading={index === 0 ? 'eager' : 'lazy'}
                decoding="async"
              />
            </article>
          );
        })}
          <div className={styles.qualitativeSpacer} aria-hidden="true" />
        </div>
        <button
          type="button"
          className={`${styles.carouselButton} ${styles.carouselButtonRight}`}
          onClick={() => scrollCarousel('next')}
          disabled={currentCase === qualitativeCases.length - 1}>
          Next
        </button>
      </div>
    </Section>
  );
}

function ContributionsSection(): ReactNode {
  return (
    <Section
      eyebrow="Contributions"
      title="Present the project in a small number of precise research-facing claims"
      lead="This section uses full-width rows rather than feature cards. Each line states one contribution and relies on spacing and separators, not framed containers."
      toneClassName={styles.toneB}>
      <div className={styles.numberedRows}>
        {contributionRows.map((item) => (
          <article key={item.title} className={styles.numberedRow}>
            <p className={styles.rowIndex}>{item.index}</p>
            <div className={styles.rowBody}>
              <h3 className={styles.rowTitle}>{item.title}</h3>
              <p className={styles.rowCopy}>{item.description}</p>
            </div>
          </article>
        ))}
      </div>
    </Section>
  );
}

function WorkflowSection(): ReactNode {
  return (
    <Section
      eyebrow="Workflow"
      title="Use one horizontal band to explain how the framework is actually used"
      lead="Each row below maps one user intent to one command and one dominant artifact, so the homepage stays clean while remaining grounded in the runnable system."
      toneClassName={styles.toneA}>
      <div className={styles.workflowRows}>
        {workflowRows.map((item) => (
          <article key={item.name} className={styles.workflowRow}>
            <div className={styles.workflowLabel}>
              <p className={styles.rowTag}>{item.name}</p>
            </div>
            <div className={styles.rowBody}>
              <h3 className={styles.rowTitle}>{item.title}</h3>
              <p className={styles.rowCopy}>{item.description}</p>
            </div>
            <div className={styles.detailBlock}>
              <span className={styles.detailLabel}>Command</span>
              <code>{item.command}</code>
            </div>
            <div className={styles.detailBlock}>
              <span className={styles.detailLabel}>Artifact</span>
              <p className={styles.detailValue}>{item.artifact}</p>
            </div>
          </article>
        ))}
      </div>
    </Section>
  );
}

function SystemSection(): ReactNode {
  return (
    <Section
      eyebrow="System"
      title="Describe the runtime structure without making the reader read source code first"
      lead="The page-level explanation stays compact: normalize inputs, resolve configuration, dispatch into the right pipeline family, and write stable artifacts."
      toneClassName={styles.toneB}>
      <div className={styles.systemGrid}>
        <div className={styles.systemColumn}>
          <p className={styles.columnLabel}>Runtime flow</p>
          <div className={styles.flowRows}>
            {runtimeFlow.map((item) => (
              <article key={item.title} className={styles.flowRow}>
                <span className={styles.stepMark}>{item.step}</span>
                <div className={styles.rowBody}>
                  <h3 className={styles.rowTitle}>{item.title}</h3>
                  <p className={styles.rowCopy}>{item.description}</p>
                </div>
              </article>
            ))}
          </div>
        </div>

        <div className={styles.systemColumn}>
          <p className={styles.columnLabel}>Pipeline families</p>
          <div className={styles.familyRows}>
            {pipelineFamilies.map((item) => (
              <article key={item.name} className={styles.familyRow}>
                <p className={styles.rowTag}>{item.name}</p>
                <p className={styles.rowCopy}>{item.description}</p>
              </article>
            ))}
          </div>
        </div>
      </div>
    </Section>
  );
}

function ResourcesSection(): ReactNode {
  return (
    <Section
      eyebrow="Resources"
      title="Keep the homepage concise and route readers into the working documentation"
      lead="The homepage explains the project; the documentation shows how to run it. These entry points separate those two jobs while keeping the overall experience continuous."
      toneClassName={styles.toneA}>
      <div className={styles.resourceRows}>
        {resourceRows.map((item) => (
          <Link key={item.title} className={styles.resourceRow} to={item.href}>
            <div className={styles.rowBody}>
              <p className={styles.rowTag}>{item.title}</p>
              <p className={styles.rowCopy}>{item.description}</p>
            </div>
            <span className={styles.resourceArrow}>Open</span>
          </Link>
        ))}
      </div>

      <div className={styles.footerBand}>
        <p className={styles.footerCopy}>
          Keep the homepage academic and declarative; let the docs carry the
          operational detail and command-level guidance.
        </p>
        <div className={styles.footerLinks}>
          <Link className={styles.heroLink} to="/docs/intro">
            Documentation Entry
          </Link>
          <Link className={styles.heroLink} to="/docs/components/overview">
            Components
          </Link>
        </div>
      </div>
    </Section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="GEditBench v2: A Human-Aligned Benchmark for General Image Editing"
      description="Academic-style project homepage for GEditBench v2, presenting the benchmark framing, workflow structure, and documentation entry points.">
      <main className={styles.page}>
        <HeroSection />
        <OverviewSection />
        <LeaderboardSection />
        <AutoPipelineSection />
        <PVCJudgeSection />
        {/*
        <ContributionsSection />
        <WorkflowSection />
        <SystemSection />
        <ResourcesSection />
        */}
        <QualitativeSection />
      </main>
    </Layout>
  );
}
