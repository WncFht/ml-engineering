import { readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { definePlumeNotesConfig } from 'vuepress-theme-plume'

const NOTES_ROOT = 'docs/notes'

function getItems(dir) {
  const fullDir = join(process.cwd(), NOTES_ROOT, dir)
  try {
    return readdirSync(fullDir)
      .map(file => {
        const fullPath = join(fullDir, file)
        const stat = statSync(fullPath)
        const link = join('/', dir, file).replace(/\\/g, '/').replace(/\.md$/, '')

        if (stat.isDirectory()) {
          const items = getItems(join(dir, file))
          if (items.length === 0) return null
          const indexFile = readdirSync(fullPath).find(f => f.toLowerCase() === 'readme.md' || f.toLowerCase() === 'index.md')
          const itemLink = indexFile ? join('/', dir, file, indexFile).replace(/\\/g, '/').replace(/\.md$/, '') : undefined
          return {
            text: formatTitle(file),
            collapsed: true,
            items: items,
            link: itemLink,
          }
        }
        else if (file.toLowerCase().endsWith('.md') && file.toLowerCase() !== 'readme.md' && file.toLowerCase() !== 'index.md') {
          return {
            text: formatTitle(file.replace(/\.md$/, '')),
            link: link,
            activeMatch: link,
          }
        }
        return null
      })
      .filter(item => item !== null)
      .sort((a, b) => {
        if (a.items && !b.items) return -1
        if (!a.items && b.items) return 1
        return a.text.localeCompare(b.text)
      })
  } catch (e) {
    return []
  }
}

function formatTitle(title) {
  return title
    .replace(/-/g, ' ')
    .replace(/\b\w/g, char => char.toUpperCase())
}

export default definePlumeNotesConfig({
  dir: 'notes',
  link: '/',
  notes: readdirSync(NOTES_ROOT)
    .filter(file => {
      const fullPath = join(NOTES_ROOT, file)
      return statSync(fullPath).isDirectory()
    })
    .map(dir => {
      const indexFile = readdirSync(join(NOTES_ROOT, dir)).find(f => f.toLowerCase() === 'readme.md' || f.toLowerCase() === 'index.md')
      const link = indexFile ? `/${dir}/${indexFile.replace(/\.md$/, '')}` : `/${dir}/`
      return {
        text: formatTitle(dir),
        dir,
        link: link,
        sidebar: [
          {
            text: formatTitle(dir),
            items: getItems(dir),
          },
        ],
      }
    }),
}) 