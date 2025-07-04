import { defineThemeConfig } from 'vuepress-theme-plume'
import notes from './notes/index.js'

// Extract the sidebar configuration from notes
const globalSidebar = notes.notes[0].sidebar

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
  // Add a global sidebar that applies to all pages
  sidebar: globalSidebar,
  footer: { message: "Machine Learning Engineering", copyright: "CC-BY-4.0" },
  plugins: {
    iconify: {
      collections: ['mdi'],
    },
    shiki: {
      languages: ["bibtex", "python", "console", "bash", "ini"],
    }
  }
})