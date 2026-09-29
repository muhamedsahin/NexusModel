import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
test("desktop, navigation, themes, search and downloads", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/tr/");
  await page.waitForTimeout(1600);
  await expect(page.locator("h1")).toContainText("Zekânın");
  await expect(page.locator("canvas")).toBeVisible();
  await page
    .getByRole("button", { name: "Katmanları ayır", exact: false })
    .click();
  await expect(
    page.getByRole("button", { name: "Birleştir", exact: false }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "Birleştir", exact: false }).click();
  for (let y = 0; y < 5000; y += 550) {
    await page.evaluate((y) => window.scrollTo(0, y), y);
    await page.waitForTimeout(90);
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(800);
  await page.screenshot({ path: "shots/home-desktop.png", fullPage: true });
  await page.getByRole("button", { name: "Temayı değiştir" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.screenshot({ path: "shots/home-light.png" });
  await page.getByRole("button", { name: "Dokümanlarda ara" }).click();
  await page
    .getByRole("textbox", { name: "Arama", exact: true })
    .fill("MicroTape");
  await page
    .locator(".search-results")
    .getByRole("link")
    .filter({ hasText: "MicroTape" })
    .first()
    .click();
  await expect(page).toHaveURL(/\/docs\//);
  await page.goto("/tr/docs/attention/");
  await page.context().grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.getByRole("button", { name: "Kodu kopyala" }).click();
  await expect(
    page.getByRole("button", { name: "Kodu kopyala" }),
  ).toContainText("Kopyalandı");
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain(
    "MultiHeadAttention",
  );
  await expect(page.locator(".math-block")).toHaveCount(2);
  await expect(page.locator(".math-block").first()).toBeVisible();
  await expect(page.locator(".katex-error")).toHaveCount(0);
  await page.getByRole("link", { name: "Switch to English" }).click();
  await expect(page).toHaveURL("/en/docs/attention/");
  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await page.screenshot({ path: "shots/docs-desktop.png", fullPage: true });
  await page.goto("/en/benchmarks/");
  await expect(page.locator(".library-bar-row")).toHaveCount(3);
  await expect(page.locator(".optimization-card")).toHaveCount(4);
  await page.getByLabel("Comparison workload").selectOption("relu_1048576");
  await page.getByRole("button", { name: "p95", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "p95", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.screenshot({ path: "shots/libraries.png", fullPage: true });
  await page.getByRole("tab", { name: "CPU vs GPU · archive" }).click();
  await page.getByLabel("Operation", { exact: true }).selectOption("LayerNorm");
  await page.getByLabel("Shape", { exact: true }).selectOption("1");
  await expect(page.locator(".comparison-result>strong")).toContainText(
    "119.43",
  );
  const dl = page.waitForEvent("download");
  await page.getByRole("button", { name: "CSV" }).click();
  expect((await dl).suggestedFilename()).toContain("gpu");
  await page.screenshot({ path: "shots/benchmarks.png", fullPage: true });
  expect(errors).toEqual([]);
});
test("published measurements and updated guide are available", async ({ request }) => {
  const response = await request.get("/benchmark-comparison.json");
  expect(response.ok()).toBe(true);
  const data = await response.json();
  expect(data.metadata.rounds).toBe(3);
  expect(data.records).toHaveLength(9);
  for (const record of data.records) {
    expect(record.results).toHaveLength(3);
    for (const result of record.results) {
      expect(result.correct).toBe(true);
      expect(result.samples_ms).toHaveLength(90);
      expect(result.median_ms).toBeGreaterThan(0);
    }
  }
  const revisions = await (await request.get("/optimization-results.json")).json();
  expect(revisions.records).toHaveLength(4);
  for (const row of revisions.records) {
    expect(row.before_samples_ms).toHaveLength(90);
    expect(row.after_samples_ms).toHaveLength(90);
    expect(row.speedup).toBeCloseTo(row.before_ms / row.after_ms, 8);
  }
  const guide = await request.get("/NexusModel-Kullanim-Kilavuzu.pdf");
  expect(guide.ok()).toBe(true);
  expect((await guide.body()).subarray(0, 5).toString()).toBe("%PDF-");
});
test("mobile overflow, menu, reading and reduced motion", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/tr/");
  await page.waitForTimeout(500);
  await expect(page.locator(".intro")).toHaveCount(0);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({ path: "shots/home-mobile.png", fullPage: true });
  await page.getByRole("button", { name: "Menüyü aç" }).click();
  await page
    .locator(".mobile-nav")
    .getByRole("link", { name: "Dokümantasyon" })
    .click();
  await page.getByRole("button", { name: "Konu dizini" }).click();
  await page
    .locator(".docs-sidebar")
    .getByRole("link", { name: "Aktivasyonlar", exact: true })
    .click();
  await expect(page.locator("h1")).toHaveText("Aktivasyonlar");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({ path: "shots/docs-mobile.png", fullPage: true });
  await page.goto("/tr/benchmarks/");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});
test("all exported pages and source references resolve", async ({
  request,
}) => {
  const routes: string[] = [];
  for (const lang of ["tr", "en"]) {
    routes.push(`/${lang}/`, `/${lang}/benchmarks/`);
    for (const slug of fs.readdirSync(path.join("out", lang, "docs")))
      routes.push(`/${lang}/docs/${slug}/`);
  }
  for (const route of routes) {
    const response = await request.get(route);
    expect(response.status(), route).toBe(200);
    const html = await response.text();
    expect(html).not.toContain("katex-error");
    for (const match of html.matchAll(/href="(\/source\/[^\"]+)"/g)) {
      expect((await request.get(match[1])).status(), match[1]).toBe(200);
    }
  }
});
