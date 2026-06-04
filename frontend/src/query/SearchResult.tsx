import { Spinner, Accordion, Form } from "react-bootstrap";
import React, { useState } from "react";

interface SearchResultProps {
  result?: string | string[];
  thinking?: string;
  isLoading?: boolean;
  componentIds?: string[];
  componentColors?: string[];
  onComponentClick?: (index: number) => void;
}

export const parseResult = (
  text: string | string[],
  componentIds?: string[],
  componentColors?: string[],
  onComponentClick?: (index: number) => void
) => {
  // Normalize: if reason is a list of steps (robot planner), join into a single string
  if (Array.isArray(text)) {
    text = text.join("\n");
  }
  const regex = /<(component|custom_bbox)_(\d+)>(.*?)<\/\1_\2>/g;
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index));
    }

    const type = match[1];
    const idStr = match[2];
    const componentName = match[3];

    const lookupId = type === "component" ? idStr : `${type}_${idStr}`;
    let index = componentIds?.findIndex((id) => String(id) === lookupId);

    if ((index === undefined || index === -1) && type === "custom_bbox") {
      // Fallback for LLMs that mistakenly use 1-based indexing
      const fallbackId = `${type}_${parseInt(idStr, 10) - 1}`;
      index = componentIds?.findIndex((id) => String(id) === fallbackId);
    }

    if (index !== undefined && index !== -1) {
      const color = componentColors?.[index] || "inherit";
      parts.push(
        <a
          key={match.index}
          href="#"
          className="text-decoration-none fw-bold"
          style={{ color }}
          onClick={(e) => {
            e.preventDefault();
            if (onComponentClick) onComponentClick(index);
          }}
        >
          {componentName}
        </a>,
      );
    } else {
      parts.push(
        <span
          key={match.index}
          className="fw-bold text-decoration-underline"
          style={{ cursor: "pointer" }}
          onClick={() =>
            console.warn(
              "Component/BBox ID not found in results list",
              lookupId,
            )
          }
        >
          {componentName}
        </span>,
      );
    }

    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }

  return parts;
};

function SearchResult({
  result,
  thinking,
  isLoading,
  componentIds,
  componentColors,
  onComponentClick,
}: SearchResultProps) {
  const [isRaw, setIsRaw] = useState(false);

  return (
    <>
      {thinking && (
        <Accordion className="mb-3" defaultActiveKey="0">
          <Accordion.Item eventKey="0">
            <Accordion.Header>
              <div className="d-flex align-items-center">
                {isLoading && (
                  <Spinner
                    animation="grow"
                    size="sm"
                    variant="secondary"
                    className="me-2"
                  />
                )}
                <span className="text-secondary">
                  {isLoading ? "Thinking..." : "Thought Process"}
                </span>
              </div>
            </Accordion.Header>
            <Accordion.Body
              style={{
                whiteSpace: "pre-line",
                maxHeight: "300px",
                overflowY: "auto",
              }}
            >
              {thinking}
            </Accordion.Body>
          </Accordion.Item>
        </Accordion>
      )}
      {result && (
        <div className="mb-2">
          <Form.Check
            type="switch"
            id="raw-mode-switch"
            label={<span className="text-muted small">Raw mode</span>}
            checked={isRaw}
            onChange={(e) => setIsRaw(e.target.checked)}
          />
        </div>
      )}
      <div className="d-flex justify-content-between align-items-start">
        <div className="flex-grow-1" style={{ minWidth: 0 }}>
          {isLoading && !result && !thinking ? (
            <div className="d-flex align-items-center gap-2 text-muted">
              <Spinner animation="border" size="sm" role="status" />
              <span>Loading...</span>
            </div>
          ) : result ? (
            <p className="mb-0 text-break" style={{ whiteSpace: "pre-line", wordBreak: "break-word" }}>
              {isRaw
                ? (Array.isArray(result) ? result.join("\n") : result)
                : parseResult(result, componentIds, componentColors, onComponentClick)}
            </p>
          ) : !thinking ? (
            <p className="text-muted mb-0">No search results yet</p>
          ) : null}
        </div>
        {(result || (isLoading && !thinking)) && (
          <span className="badge bg-secondary ms-3">Result</span>
        )}
      </div>
    </>
  );
}

export default SearchResult;
