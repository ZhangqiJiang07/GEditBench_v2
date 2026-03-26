import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/environment-setup',
        'getting-started/quickstart-autopipeline',
        'tutorials/first-annotation',
        'tutorials/first-eval',
        'tutorials/first-train-pairs',
      ],
    },
    {
      type: 'category',
      label: 'Pipeline Guides',
      items: [
        {
          type: 'doc',
          id: 'tutorial/guide-for-pipelines/overview',
          label: 'How to Choose a Pipeline',
        },
        'tutorial/guide-for-pipelines/object-centric',
        'tutorial/guide-for-pipelines/human-centric',
        'tutorial/guide-for-pipelines/vlm-as-a-judge',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'tutorial/framework-overview',
        'concepts/config-system',
      ],
    },
    {
      type: 'category',
      label: 'Components',
      items: [
        'components/overview',
        {
          type: 'category',
          label: 'Modules',
          items: [
            'components/modules/overview',
            'components/modules/parser-grounder-pipe',
            'components/modules/judge-modules',
            'components/modules/clip-pipe',
            'components/modules/dino-v3-pipe',
            'components/modules/lpips-pipe',
            'components/modules/ssim-pipe',
            'components/modules/sam-pipe',
            'components/modules/depth-anything-v2-pipe',
            'components/modules/face-pipes',
            'components/modules/hair-consistency-pipe',
            'components/modules/body-pipes',
          ],
        },
        {
          type: 'category',
          label: 'Primitives',
          items: [
            'components/primitives/overview',
            {
              type: 'category',
              label: 'Clients',
              items: [
                'components/primitives/clients/base-client',
                'components/primitives/clients/openai-client',
                'components/primitives/clients/google-client',
                'components/primitives/clients/vllm-client',
              ],
            },
            'components/primitives/prompt-adapters',
            {
              type: 'category',
              label: 'Vision and Analysis',
              items: [
                'components/primitives/face-analyzer',
                'components/primitives/grounding-dino',
                'components/primitives/human-segmenter',
                'components/primitives/human-skeleton',
              ],
            },
            {
              type: 'category',
              label: 'Similarity and Masking',
              items: [
                'components/primitives/mask-processor',
                'components/primitives/semantic-consistency',
                'components/primitives/visual-consistency',
                'components/primitives/ssim-utilities',
              ],
            },
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Extending AutoPipeline',
      items: [
        'extending/overview',
        'extending/add-primitive',
        'extending/add-module',
        'extending/add-pipeline',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/cli-autopipeline',
        'reference/output-formats',
        'reference/task-support-matrix',
        'reference/config-reference',
        'reference/pipeline-config-pattern',
        {
          type: 'doc',
          id: 'troubleshooting/common-issues',
          label: 'Common Issues',
        },
      ],
    },
  ],
};

export default sidebars;
