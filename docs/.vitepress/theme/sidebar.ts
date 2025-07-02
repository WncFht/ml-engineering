import { readdirSync, statSync } from 'node:fs';
import { join, resolve } from 'node:path';

const DOCS_ROOT = resolve(__dirname, '..', '..')

function getSidebarItems(dir, currentRoot) {
  const fullDir = join(currentRoot, dir)
  try {
    return readdirSync(fullDir)
      .filter(file => file !== 'index.md' && file !== 'images' && !file.endsWith('.html') && file !== 'README.md')
      .map(file => {
        const fullPath = join(fullDir, file)
        const stat = statSync(fullPath)
        const link = join('/', dir, file).replace(/\\/g, '/').replace(/\.md$/, '')

        if (stat.isDirectory()) {
          const items = getSidebarItems(join(dir, file), currentRoot)
          if (items.length === 0) return null;

          const indexFile = readdirSync(fullPath).find(f => f.toLowerCase() === 'readme.md' || f.toLowerCase() === 'index.md');
          const groupLink = indexFile ? join('/', dir, file, indexFile).replace(/\\/g, '/').replace(/\.md$/, '') : undefined;

          return {
            text: formatTitle(file),
            collapsible: true,
            items: items,
            ...(groupLink && { link: groupLink })
          }
        } else {
          return {
            text: formatTitle(file.replace(/\.md$/, '')),
            link: link
          }
        }
      })
      .filter(item => item !== null)
      .sort((a, b) => {
          if (a.items && !b.items) return -1;
          if (!a.items && b.items) return 1;
          return a.text.localeCompare(b.text);
      });
  } catch (e) {
    console.error(`Could not read directory: ${fullDir}`, e);
    return [];
  }
}

function formatTitle(title) {
  return title
    .replace(/-/g, ' ')
    .replace(/\b\w/g, char => char.toUpperCase());
}

export function getSidebar() {
    const sidebar = {}
    const topLevelDirs = readdirSync(DOCS_ROOT).filter(file => {
        const fullPath = join(DOCS_ROOT, file)
        return statSync(fullPath).isDirectory() && file !== '.vitepress' && file !== 'images'
    })

    for (const dir of topLevelDirs) {
        const items = getSidebarItems(dir, DOCS_ROOT);
        if (items.length > 0) {
            sidebar[`/${dir}/`] = [{
                text: formatTitle(dir),
                items: items
            }]
        }
    }
    return sidebar
}

export function getNav() {
    const nav = [{ text: 'Home', link: '/' }]
    const topLevelDirs = readdirSync(DOCS_ROOT).filter(file => {
        const fullPath = join(DOCS_ROOT, file)
        try {
            const hasContent = statSync(fullPath).isDirectory() && 
                               (readdirSync(fullPath).some(f => f.toLowerCase().endsWith('.md')));
            return hasContent && file !== '.vitepress' && file !== 'images' && file !== 'theme'
        } catch(e) {
            return false;
        }
    })
    .sort()

    for (const dir of topLevelDirs) {
        const indexFile = readdirSync(join(DOCS_ROOT, dir)).find(f => f.toLowerCase() === 'readme.md' || f.toLowerCase() === 'index.md');
        const link = indexFile ? `/${dir}/${indexFile.replace(/\.md$/, '')}` : `/${dir}/`;
        
        nav.push({
            text: formatTitle(dir),
            link: link
        })
    }
    return nav
} 