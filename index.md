# GenoMAS Project Page

This directory contains the project website for the GenoMAS paper: "A Multi-Agent Framework for Scientific Discovery via Code-Driven Gene Expression Analysis"

## 🌐 Website Structure

```
.
├── index.html          # Main HTML file
├── styles.css          # Stylesheet with modern design
├── script.js           # JavaScript for interactivity
├── imgs/              # Image assets
│   ├── logo.png
│   ├── system_diagram.png
│   ├── Programming_agent.png
│   ├── Main_result_table.jpg
│   ├── Individual_task_bar_plot.jpg
│   └── Agent_collaboration_patterns.jpg
└── PROJECT_PAGE_README.md  # This file
```

## 🚀 Deployment to GitHub Pages

### Option 1: Deploy from this Branch

1. Push this branch to GitHub:
   ```bash
   git add index.html styles.css script.js PROJECT_PAGE_README.md
   git commit -m "Add GenoMAS project website"
   git push origin project-page
   ```

2. Go to your GitHub repository settings
3. Navigate to **Pages** in the left sidebar
4. Under **Source**, select the `project-page` branch
5. Select the root directory `/`
6. Click **Save**
7. Your site will be published at `https://<username>.github.io/GenoMAS/`

### Option 2: Deploy using gh-pages Branch

1. Create a dedicated gh-pages branch:
   ```bash
   # From the project-page branch
   git checkout -b gh-pages
   git push origin gh-pages
   ```

2. In GitHub repository settings > Pages:
   - Select the `gh-pages` branch
   - Select root directory
   - Save

### Option 3: Use GitHub Actions (Automated)

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ project-page ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./
          publish_branch: gh-pages
```

## 🖥️ Local Development

To view the website locally:

1. **Simple Python Server:**
   ```bash
   cd /home/techt/Desktop/GenoMAS2
   python3 -m http.server 8000
   ```
   Then open `http://localhost:8000` in your browser

2. **Using Live Server (VS Code Extension):**
   - Install the "Live Server" extension in VS Code
   - Right-click on `index.html`
   - Select "Open with Live Server"

3. **Direct File Opening:**
   - Simply double-click `index.html`
   - Or drag it into your browser
   - Note: Some features may not work without a server

## ✏️ Customization

### Update arXiv Link
Once your paper is on arXiv, update the placeholder links in `index.html`:
- Find `https://arxiv.org/abs/XXXX.XXXXX`
- Replace with your actual arXiv URL

### Update Citation
Update the BibTeX citation in:
- `index.html` (citation box)
- `script.js` (copyCitation function)

### Modify Colors
Edit CSS variables in `styles.css`:
```css
:root {
    --primary-color: #2563eb;    /* Main brand color */
    --secondary-color: #10b981;  /* Accent color */
    --accent-color: #f59e0b;     /* Highlight color */
}
```

### Add or Remove Sections
Edit `index.html` to add/remove sections as needed. Update navigation links in the navbar accordingly.

## 📱 Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Smooth Scrolling**: Animated navigation between sections
- **Interactive Elements**: Hover effects, fade-in animations
- **Copy Citation**: One-click BibTeX citation copying
- **Back to Top Button**: Appears when scrolling down
- **Modern UI**: Clean, professional design with gradient accents

## 🎨 Design Philosophy

The website follows modern web design principles:
- Clean, minimalist layout
- Hierarchy through typography and spacing
- Gradient accents for visual interest
- Card-based components for content organization
- Smooth animations for better UX

## 🔧 Browser Compatibility

Tested and working on:
- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers

## 📝 Content Sections

1. **Hero**: Title, authors, affiliations, and main CTAs
2. **Abstract**: Paper summary with key metrics
3. **Key Contributions**: Four main contributions in card format
4. **Method Overview**: System architecture and agent workflow
5. **Results**: Performance comparison and ablation studies
6. **Collaboration Patterns**: Agent interaction analysis
7. **Paper & Citation**: Links and BibTeX citation
8. **Team**: Author information and acknowledgments

## 🛠️ Technologies Used

- **HTML5**: Semantic markup
- **CSS3**: Modern styling with CSS Grid and Flexbox
- **JavaScript (Vanilla)**: Interactive features without dependencies
- **Google Fonts**: Inter and JetBrains Mono
- **SVG Icons**: Inline icons for better performance

## 📄 License

The website code is part of the GenoMAS project. Please refer to the main repository LICENSE for details.

## 🤝 Contributing

If you find any issues or have suggestions:
1. Open an issue in the GitHub repository
2. Submit a pull request with improvements
3. Contact the authors directly

## 📧 Contact

For questions about the website:
- Haoyang Liu: hl57@illinois.edu
- Yijiang Li: yijiangli@ucsd.edu
- Haohan Wang: haohanw@illinois.edu

---

Built with ❤️ for the scientific community
