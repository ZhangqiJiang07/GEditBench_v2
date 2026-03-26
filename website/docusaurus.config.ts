import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'GEditBench v2',
  tagline: 'Open-source benchmark and pipeline docs for image-edit evaluation',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://zhangqijiang07.github.io',
  baseUrl: '/GEditBench_v2/',
  organizationName: 'ZhangqiJiang07',
  projectName: 'GEditBench_v2',

  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'docs',
          editUrl:
            'https://github.com/ZhangqiJiang07/GEditBench_v2/tree/main/website/',
          showLastUpdateTime: true,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: false,
    },
    navbar: {
      title: 'GEditBench v2',
      logo: {
        alt: 'GEditBench v2 Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          to: '/docs/getting-started/quickstart-autopipeline',
          label: 'Quickstart',
          position: 'left',
        },
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Pipeline Docs',
        },
        {
          href: 'https://github.com/ZhangqiJiang07/GEditBench_v2',
          label: 'Repo',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'light',
      links: [
        {
          title: 'Get Started',
          items: [
            {
              label: 'Overview',
              to: '/docs/intro',
            },
            {
              label: 'Quickstart',
              to: '/docs/getting-started/quickstart-autopipeline',
            },
          ],
        },
        {
          title: 'Workflows',
          items: [
            {
              label: 'Annotation',
              to: '/docs/tutorials/first-annotation',
            },
            {
              label: 'Evaluation',
              to: '/docs/tutorials/first-eval',
            },
          ],
        },
        {
          title: 'Reference',
          items: [
            {
              label: 'CLI',
              to: '/docs/reference/cli-autopipeline',
            },
            {
              label: 'Output Formats',
              to: '/docs/reference/output-formats',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} GEditBench Contributors.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
