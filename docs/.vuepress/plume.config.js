import { defineThemeConfig } from 'vuepress-theme-plume'
import notes from './notes/index.js'

export default defineThemeConfig({
  lang: 'zh-CN',
  blog: false,
  social: [
    { icon: 'github', link: 'https://github.com/stas00/ml-engineering' },
  ],
  navbar: [
    { text: 'Home', link: '/' },
  ],
  notes,
  footer: { message: "Machine Learning Engineering", copyright: "CC-BY-4.0" },
})