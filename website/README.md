# Website

This site is built with [Docusaurus](https://docusaurus.io/) and deployed as the GitHub Pages site for the `GEditBench_v2` repository.

## Requirements

- Node.js 20 or newer
- npm

The repository includes `.nvmrc`, so if you use `nvm` you can run:

```bash
nvm use
```

## Install Dependencies

```bash
npm ci
```

## Local Development

```bash
npm run start
```

## Production Build

```bash
npm run build
```

The static output will be generated in `build/`.

## GitHub Pages Deployment

Deployment is handled by the repository workflow at `../.github/workflows/deploy-pages.yml`.

- Push changes under `website/` to `main`
- GitHub Actions builds the site
- The workflow publishes the `website/build` artifact to GitHub Pages

The expected public URL is:

```text
https://zhangqijiang07.github.io/GEditBench_v2/
```
