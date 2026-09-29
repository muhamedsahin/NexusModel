export type Locale = "tr" | "en";
export type Text = { tr: string; en: string };
export const tx = (tr: string, en: string): Text => ({ tr, en });
export type Section = {
  id: string;
  title: Text;
  body: Text;
  code?: string;
  formula?: string;
  table?: { headers: Text[]; rows: string[][] };
};
export type Article = {
  slug: string;
  group: number;
  title: Text;
  description: Text;
  source: string;
  sections: Section[];
};
export const s = (
  id: string,
  tr: string,
  en: string,
  bodyTr: string,
  bodyEn: string,
  extra: Partial<Section> = {},
): Section => ({ id, title: tx(tr, en), body: tx(bodyTr, bodyEn), ...extra });
