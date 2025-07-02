import { viteBundler } from '@vuepress/bundler-vite'
import { defineUserConfig } from 'vuepress'
import { plumeTheme } from 'vuepress-theme-plume'
import themeConfig from './plume.config.js'

export default defineUserConfig({
  base: '/ml-engineering/',
  // 请不要忘记设置默认语言
  blog: false,
  lang: 'zh-CN',
  head: [
    [
        'link', { rel: 'icon', href: '/images/logo.png' },
    ]
  ],
  locales: {
    '/': { lang: 'zh-CN', title: 'Machine Learning Engineering' }
  },
  theme: plumeTheme(themeConfig),
  bundler: viteBundler(),
})
