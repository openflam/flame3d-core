import type * as React from "react";

// Define BoundingBox interface
export interface BoundingBox {
  corners: [number, number, number][];
}

// Define Route type - list of 3D coordinates
export type Route = [number, number, number][];

// Define SearchQuery interface
export interface SearchQueryItem {
  type: "text" | "image";
  value: string;
}

export type SearchQuery = SearchQueryItem[];

// Define Component interface (matches API response)
export interface Component {
  bbox: BoundingBox;
  caption: string;
  component_id: string;
}

// Define SearchResult interface (matches API response)
export interface SearchResult {
  reason: string;
  search_time_ms: number;
  components: Component[];
}

// Define custom attributes for babylon-viewer element
interface HTML3DElementAttributes extends React.DetailedHTMLProps<
  React.HTMLAttributes<HTMLElement>,
  HTMLElement
> {
  source?: string;
  environment?: string;
}

declare global {
  namespace JSX {
    interface IntrinsicElements {
      "babylon-viewer": HTML3DElementAttributes;
    }
  }
}

export {};
