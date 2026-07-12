import re
with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

style_match = re.search(r'<style>(.*?)</style>', content, re.DOTALL)
if style_match:
    with open('style.css', 'w', encoding='utf-8') as f:
        f.write(style_match.group(1).strip())

script_match = re.search(r'<script>(.*?)</script>', content, re.DOTALL)
if script_match:
    with open('script.js', 'w', encoding='utf-8') as f:
        f.write(script_match.group(1).strip())

html_content = re.sub(r'<style>.*?</style>', '<link rel="stylesheet" href="style.css">', content, flags=re.DOTALL)
html_content = re.sub(r'<script>.*?</script>', '<script src="script.js"></script>', html_content, flags=re.DOTALL)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)
