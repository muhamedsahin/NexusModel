import { chromium } from "playwright";
import path from "node:path";
import { pathToFileURL } from "node:url";

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage();
  await page.goto(pathToFileURL(path.resolve("kilavuz.html")).href);
  await page.pdf({
    path: path.resolve("NexusModel-Kullanim-Kilavuzu.pdf"),
    printBackground: true,
    preferCSSPageSize: true,
  });
} finally {
  await browser.close();
}
console.log("PDF guide rendered from kilavuz.html.");
