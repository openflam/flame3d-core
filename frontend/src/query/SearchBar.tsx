import {
  InputGroup,
  FormControl,
  Button,
  Form,
  ProgressBar,
} from "react-bootstrap";
import { useState } from "react";
import type { BoundingBox, SearchQuery } from "./types/global";
import { downloadAllComponents } from "./query";
import { worldToViewerBbox } from "./transforms";

// react-bootstrap's polymorphic Button can blow up TS's union inference
// (TS2590) when nested inside InputGroup under strict mode. This thin alias
// keeps the same runtime component while sidestepping the inference explosion.
const Btn = Button as React.ElementType;

interface SearchBarProps {
  onSearch: (searchQuery: SearchQuery, modelName?: string) => void;
  onAnnotationsDownloaded: (
    bboxes: BoundingBox[],
    annotations: string[],
  ) => void;
  showAutoTags: boolean;
  onShowAutoTagsChange: (show: boolean) => void;
  showOccupancyGrid: boolean;
  onShowOccupancyGridChange: (show: boolean) => void;
  datasetName: string;
}

function SearchBar({
  onSearch,
  onAnnotationsDownloaded,
  showAutoTags,
  onShowAutoTagsChange,
  showOccupancyGrid,
  onShowOccupancyGridChange,
  datasetName,
}: SearchBarProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [modelName, setModelName] = useState("");
  const [isDownloadingAnnotations, setIsDownloadingAnnotations] =
    useState(false);
  const [annotationsLoaded, setAnnotationsLoaded] = useState(false);

  const handleDownloadAnnotations = async () => {
    setIsDownloadingAnnotations(true);
    try {
      const data = await downloadAllComponents(datasetName);
      // Components come from the server in the world (Z-up) frame.
      const bboxes: BoundingBox[] = data.map((item: any) =>
        worldToViewerBbox(item.bbox),
      );
      const annotationList: string[] = data.map((item) =>
        item.connected_comp_id.toString(),
      );
      onAnnotationsDownloaded(bboxes, annotationList);
      setAnnotationsLoaded(true);
    } catch (error) {
      console.error("Failed to download annotations:", error);
    } finally {
      setIsDownloadingAnnotations(false);
    }
  };

  const handleSearch = () => {
    if (searchTerm.trim()) {
      const searchQuery: SearchQuery = [{ type: "text", value: searchTerm }];
      onSearch(searchQuery, modelName);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <>
      <InputGroup className="mb-3">
        <FormControl
          placeholder="Model (Default: gpt-5.4)"
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          style={{ maxWidth: "200px" }}
        />

        <FormControl
          placeholder="Ask..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          onKeyDown={handleKeyPress}
        />

        <Btn variant="outline-primary" onClick={handleSearch}>
          Ask
        </Btn>
      </InputGroup>

      <div className="d-flex align-items-center justify-content-between mb-3">
        <div className="d-flex align-items-center gap-3">
          {isDownloadingAnnotations ? (
            <ProgressBar
              animated
              striped
              now={100}
              label="Downloading…"
              style={{ width: "160px", height: "31px" }}
            />
          ) : (
            <Btn
              variant={annotationsLoaded ? "secondary" : "outline-secondary"}
              disabled={annotationsLoaded}
              onClick={handleDownloadAnnotations}
            >
              {annotationsLoaded
                ? "Annotations Downloaded"
                : "Download Annotations"}
            </Btn>
          )}
          <Form.Check
            type="checkbox"
            label="Show Auto Tags"
            checked={showAutoTags}
            disabled={!annotationsLoaded}
            onChange={(e) => onShowAutoTagsChange(e.target.checked)}
          />
          <Form.Check
            type="checkbox"
            label="Show Occupancy Grid"
            checked={showOccupancyGrid}
            onChange={(e) => onShowOccupancyGridChange(e.target.checked)}
          />
        </div>
      </div>
    </>
  );
}

export default SearchBar;
