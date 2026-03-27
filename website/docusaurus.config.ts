import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'AutoPipeline',
  tagline: 'Open-source pipelines for image-edit evaluation and preference data construction',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://autopipeline-docs.example.com',
  baseUrl: '/',
  organizationName: 'open-edit',
  projectName: 'autopipeline-docs',

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
            'https://gitlab.basemind.com/i-jiangzhangqi/open_edit/-/tree/main',
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
      title: 'AutoPipeline',
      logo: {
        alt: 'AutoPipeline Logo',
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
          href: 'https://gitlab.basemind.com/i-jiangzhangqi/open_edit',
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
      copyright: `Copyright © ${new Date().getFullYear()} AutoPipeline Contributors.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
