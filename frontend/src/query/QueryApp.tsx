import { styles } from "./appStyles";
import { useState, useEffect } from "react";

import "bootstrap/dist/css/bootstrap.min.css";

import SearchBar from "./SearchBar";
import Model3DViewer from "./Model3DViewer";
import SearchResult from "./SearchResult";
import SearchComponentList from "./SearchComponentList";
import { queryStream, SEARCH_SERVER_URL } from "./query";
import type { BoundingBox, SearchQuery, Route } from "./types/global";

interface QueryAppProps {
  /** Dataset to query — passed in from the surrounding app navigation. */
  datasetName: string;
  /** Return to the dataset list. */
  onBack: () => void;
}

function QueryApp({ datasetName, onBack }: QueryAppProps) {
  const [boundingBox, setBoundingBox] = useState<BoundingBox[]>([]);
  const [captions, setCaptions] = useState<string[]>([]);
  const [componentIds, setComponentIds] = useState<string[]>([]);
  const [componentColors, setComponentColors] = useState<string[]>([]);
  const [showAutoTags, setShowAutoTags] = useState(false);
  const [showOccupancyGrid, setShowOccupancyGrid] = useState(false);
  const [autoTagBBoxes, setAutoTagBBoxes] = useState<BoundingBox[]>([]);
  const [occupancyGrid, setOccupancyGrid] = useState<BoundingBox[]>([]);
  const [annotations, setAnnotations] = useState<string[]>([]);
  const [searchResult, setSearchResult] = useState<string | string[] | undefined>(
    undefined,
  );
  const [thinking, setThinking] = useState<string | undefined>(undefined);
  const [isLoading, setIsLoading] = useState(false);
  const [route, setRoute] = useState<Route>([]);
  const [focusedComponentIndex, setFocusedComponentIndex] = useState<
    number | null
  >(null);
  const [isLeftColumnCollapsed, setIsLeftColumnCollapsed] = useState(false);

  const handleAnnotationsDownloaded = (
    bboxes: BoundingBox[],
    annotationList: string[],
  ) => {
    setAutoTagBBoxes(bboxes);
    setAnnotations(annotationList);
  };

  // Load occupancy_bbox.json on mount
  useEffect(() => {
    fetch("/data/occupancy_bbox.json")
      .then((response) => response.json())
      .then((data) => {
        // Extract bounding boxes from the JSON data.
        // Swap axes as needed -- results of trial and error.
        const bboxes: BoundingBox[] = data.map((item: any) => ({
          corners: item.bbox.corners.map((c: number[]) => [c[1], c[2], c[0]]),
        }));
        setOccupancyGrid(bboxes);
      })
      .catch((error) => {
        console.error("Error loading occupancy_bbox.json:", error);
      });
  }, []);

  const handleSearch = async (searchQuery: SearchQuery, modelName?: string) => {
    setIsLoading(true);
    setSearchResult(undefined);
    setThinking(undefined);
    setFocusedComponentIndex(null);
    setRoute([]); // Clear any existing route

    const handleResult = (result: any) => {
      const dbComponents = result.components || [];
      const customBBoxes = result.custom_bboxes || [];

      const allBBoxes = [
        ...dbComponents.map((c: any) => c.bbox),
        ...customBBoxes
      ];

      const allCaptions = [
        ...dbComponents.map((c: any) => c.caption),
        ...customBBoxes.map(() => "Custom Location")
      ];

      const allComponentIds = [
        ...dbComponents.map((c: any) => String(c.component_id)),
        ...customBBoxes.map((_: any, i: number) => `custom_bbox_${i}`)
      ];

      const allColors = allBBoxes.map(
        (_: any, i: number) => `hsl(${(i * 137.508) % 360}, 70%, 50%)`,
      );

      setBoundingBox(allBBoxes);
      setCaptions(allCaptions);
      setComponentIds(allComponentIds);
      setComponentColors(allColors);
      setSearchResult(result.reason);
    };

    let currentThinking = "";
    try {
      await queryStream(
        searchQuery,
        "Tools",
        datasetName,
        modelName,
        undefined,
        (event) => {
          if (event.type === "thinking") {
            currentThinking += event.content;
            setThinking(currentThinking);
          } else if (event.type === "result") {
            handleResult(event.data);
          } else if (event.type === "error") {
            console.error("Stream error:", event.error);
            setSearchResult(`Error: ${event.error}`);
          }
        },
      );
    } catch (e) {
      console.error("Query stream failed:", e);
      setSearchResult("Search failed due to a network or server error.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={styles.rootContainer}>
      {/* Top Search Bar Area */}
      <div style={styles.topBarArea}>
        <div className="d-flex align-items-center gap-2 mb-1">
          <button
            onClick={onBack}
            className="btn btn-link btn-sm p-0 text-decoration-none"
            title="Back to datasets"
            style={{ fontSize: "0.8rem", lineHeight: 1 }}
          >
            ← Back
          </button>
          <span style={{ fontWeight: 600, fontSize: "0.8rem", color: "#6c757d" }}>
            {datasetName}
          </span>
        </div>
        <SearchBar
          onSearch={handleSearch}
          onAnnotationsDownloaded={handleAnnotationsDownloaded}
          showAutoTags={showAutoTags}
          onShowAutoTagsChange={setShowAutoTags}
          showOccupancyGrid={showOccupancyGrid}
          onShowOccupancyGridChange={setShowOccupancyGrid}
          datasetName={datasetName}
        />
      </div>

      {/* Main Content Area */}
      <div style={styles.mainContentArea}>
        {/* Collapsible Left Column */}
        <div
          style={{
            ...styles.leftColumnBase,
            width: isLeftColumnCollapsed ? "0px" : "460px",
            borderRight: isLeftColumnCollapsed ? "none" : "1px solid #dee2e6",
          }}
        >
          {/* Search Result Box */}
          <div style={styles.searchResultBox}>
            <SearchResult
              result={searchResult}
              thinking={thinking}
              isLoading={isLoading}
              componentIds={componentIds}
              componentColors={componentColors}
              onComponentClick={(i: number) => setFocusedComponentIndex(i)}
            />
          </div>

          {/* Search Component List */}
          <div style={styles.searchComponentListWrapper}>
            <SearchComponentList
              componentIds={componentIds}
              captions={captions}
              datasetName={datasetName}
              onComponentClick={(i) => setFocusedComponentIndex(i)}
              focusedComponentIndex={focusedComponentIndex}
            />
          </div>
        </div>

        {/* Collapse Toggle Button */}
        <button
          onClick={() => setIsLeftColumnCollapsed(!isLeftColumnCollapsed)}
          style={styles.collapseToggleButton}
          title={isLeftColumnCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          {isLeftColumnCollapsed ? "▶" : "◀"}
        </button>

        {/* 3D Viewer */}
        <div style={styles.viewerContainer}>
          <Model3DViewer
            source={`${SEARCH_SERVER_URL}/load_mesh?dataset_name=${encodeURIComponent(datasetName)}`}
            boundingBox={boundingBox}
            captions={captions}
            componentIds={componentIds}
            componentColors={componentColors}
            autoTagBBoxes={autoTagBBoxes}
            showAutoTags={showAutoTags}
            occupancyGrid={occupancyGrid}
            showOccupancyGrid={showOccupancyGrid}
            annotations={annotations}
            route={route}
            datasetName={datasetName}
            focusedBBoxIndex={focusedComponentIndex}
            externalSelectedBBoxIndex={focusedComponentIndex}
          />
        </div>
      </div>
    </div>
  );
}

export default QueryApp;
