import { defineConfig } from 'vitepress'
import { getNav, getSidebar } from './theme/sidebar'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Machine Learning Engineering",
  description: "A book on Machine Learning Engineering",
  base: '/ml-engineering/',
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: getNav(),
    sidebar: getSidebar(),
    socialLinks: [
      { icon: 'github', link: 'https://github.com/stas00/ml-engineering' }
    ]
  },
  srcDir: './docs'
}) 