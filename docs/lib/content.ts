import { tx } from "./types";
import { basics } from "./content-basics";
import { layers } from "./content-layers";
import { systems } from "./content-systems";
import { reference } from "./content-reference";
import { comparisonGuide } from "./content-comparison";
import { performanceGuide } from "./content-performance";
export * from "./types";
export { quickCode } from "./quick-code";
export const groups = [
  tx("Başlangıç", "Getting started"),
  tx("Çekirdek mimari", "Core architecture"),
  tx("Katman referansı", "Layer reference"),
  tx("Sistem ve performans", "Systems & performance"),
];
export const articles = [
  ...basics,
  ...layers,
  ...systems,
  ...reference,
  comparisonGuide,
  performanceGuide,
].sort((a, b) => a.group - b.group);
export const getArticle = (slug: string) =>
  articles.find((a) => a.slug === slug);
