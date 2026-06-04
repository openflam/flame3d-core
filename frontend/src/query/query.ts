import type { SearchResult, SearchQuery } from "./types/global";

// All query/search endpoints are served by the Flask backend under `/api`,
// which the Vite dev server proxies (see vite.config.ts). Using a relative
// base keeps it working both in docker-compose and standalone dev.
export const SEARCH_SERVER_URL = "/api";

export async function getProvidersList(): Promise<string[]> {
  const response = await fetch(`${SEARCH_SERVER_URL}/get_providers_list`, {
    method: "GET",
  });
  if (!response.ok) {
    throw new Error(
      `Get providers list failed: ${response.status} ${response.statusText}`,
    );
  }
  const data = await response.json();
  return data.providers as string[];
}

export async function query(
  searchQuery: SearchQuery,
  method: string,
  datasetName: string,
  modelName?: string,
): Promise<SearchResult> {
  console.log("Querying with:", searchQuery, "using method:", method);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: datasetName,
        query: searchQuery,
        method: method,
        model_name: modelName,
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Search request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    console.log("Received response:", data);

    // Return the data as-is from the API
    // Server returns: {"reason": "...", "search_time_ms": ..., "components": [{bbox: {...}, caption: "..."}]}
    return data as SearchResult;
  } catch (error) {
    console.error("Error querying search server:", error);
    throw error;
  }
}

export async function queryStream(
  searchQuery: SearchQuery,
  method: string,
  datasetName: string,
  modelName: string | undefined,
  mode: string | undefined,
  onEvent: (eventData: any) => void
): Promise<void> {
  const route = mode === "robot" ? "/robot_steps" : "/search_stream";
  console.log("Streaming query with:", searchQuery, "using method:", method, "route:", route);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}${route}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: datasetName,
        query: searchQuery,
        method: method,
        model_name: modelName,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(
        `Search stream request failed: ${response.status} ${response.statusText}`
      );
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");

      buffer = lines.pop() || ""; // Keep the last incomplete line in the buffer

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const dataStr = line.slice(6);
          try {
            const data = JSON.parse(dataStr);
            onEvent(data);
          } catch (e) {
            console.error("Error parsing stream data:", e, dataStr);
          }
        }
      }
    }
  } catch (error) {
    console.error("Error streaming search server:", error);
    throw error;
  }
}

export async function queryDirections(
  source: SearchQuery,
  destination: SearchQuery,
  method: string,
  datasetName: string,
  modelName?: string,
): Promise<{
  path: number[][];
  source_bbox: any;
  destination_bbox: any;
  source_reason: string;
  destination_reason: string;
}> {
  console.log(
    "Querying directions from:",
    source,
    "to:",
    destination,
    "using method:",
    method,
  );

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/get_route`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: datasetName,
        source: source,
        destination: destination,
        method: method,
        model_name: modelName,
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Directions request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();

    return data;
  } catch (error) {
    console.error("Error querying directions:", error);
    throw error;
  }
}

export async function getComponentInfo(
  componentId: string,
  datasetName: string,
): Promise<{
  component_id: string;
  caption: string;
  image_name: string;
  image_base64: string | null;
  fraction_visible: number;
  image_width: number;
  image_height: number;
}> {
  console.log("Fetching component info for:", componentId);

  try {
    const response = await fetch(
      `${SEARCH_SERVER_URL}/get_component_info?dataset_name=${encodeURIComponent(datasetName)}&component_id=${encodeURIComponent(componentId)}`,
      {
        method: "GET",
      },
    );

    if (!response.ok) {
      throw new Error(
        `Component info request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    console.log("Received component info:", data);

    return data;
  } catch (error) {
    console.error("Error fetching component info:", error);
    throw error;
  }
}

export async function deleteComponent(
  componentId: string,
  datasetName: string,
): Promise<{ component_id: string; deleted: boolean }> {
  console.log("Deleting component:", componentId);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/delete_component`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: datasetName,
        component_id: componentId,
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Delete component request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    console.log("Deleted component:", data);

    return data;
  } catch (error) {
    console.error("Error deleting component:", error);
    throw error;
  }
}

export async function downloadAllComponents(
  datasetName: string,
): Promise<
  Array<{ connected_comp_id: number; bbox: { corners: [number, number, number][] } }>
> {
  console.log("Downloading all components...");

  try {
    const response = await fetch(
      `${SEARCH_SERVER_URL}/download_all_components?dataset_name=${encodeURIComponent(datasetName)}`,
      { method: "GET" },
    );

    if (!response.ok) {
      throw new Error(
        `Download failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    console.log(`Downloaded ${data.length} components`);
    return data;
  } catch (error) {
    console.error("Error downloading all components:", error);
    throw error;
  }
}

export async function updateComponent(
  componentId: string,
  updates: { caption?: string; bbox?: { corners: [number, number, number][] } },
  datasetName: string,
): Promise<{ component_id: string; caption?: string; bbox?: object }> {
  console.log("Updating component:", componentId, updates);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/update_component`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: datasetName,
        component_id: componentId,
        ...updates,
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Update component request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    console.log("Updated component:", data);

    return data;
  } catch (error) {
    console.error("Error updating component:", error);
    throw error;
  }
}

export async function addComponent(
  datasetName: string,
  caption: string,
  bbox: { corners: [number, number, number][] },
  imageBase64: string | null = null
): Promise<{ component_id: string; caption: string; bbox: object; best_crop: string; added: boolean }> {
  console.log("Adding component:", datasetName, caption);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/add_component`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: datasetName,
        caption: caption,
        bbox: bbox,
        image_base64: imageBase64,
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Add component request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    console.log("Added component:", data);

    return data;
  } catch (error) {
    console.error("Error adding component:", error);
    throw error;
  }
}

export async function callTool(
  toolName: string,
  args: any,
  datasetName: string,
): Promise<any> {
  console.log("Calling tool:", toolName, args);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/call_tool`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: datasetName,
        tool_name: toolName,
        arguments: args,
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Call tool request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    console.log("Tool result:", data);

    return data;
  } catch (error) {
    console.error("Error calling tool:", error);
    throw error;
  }
}

export async function saveBenchmark(
  datasetName: string,
  question: string,
  expectedAnswer: string,
  benchmarkName: string,
  benchmarkType: string,
): Promise<any> {
  console.log("Saving benchmark:", { datasetName, benchmarkName, benchmarkType });

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/save_benchmark`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: datasetName,
        question: question,
        expected_answer: expectedAnswer,
        benchmark_name: benchmarkName,
        benchmark_type: benchmarkType,
      }),

    });

    if (!response.ok) {
      throw new Error(
        `Save benchmark request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    console.log("Saved benchmark:", data);

    return data;
  } catch (error) {
    console.error("Error saving benchmark:", error);
    throw error;
  }
}

export async function getBenchmarkNames(): Promise<string[]> {
  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/get_benchmark_names`);
    if (!response.ok) throw new Error("Failed to fetch benchmark names");
    const data = await response.json();
    return data.names || [];
  } catch (error) {
    console.error("Error fetching benchmark names:", error);
    return [];
  }
}

export async function getBenchmark(name: string): Promise<any> {
  try {
    const response = await fetch(
      `${SEARCH_SERVER_URL}/get_benchmark?name=${encodeURIComponent(name)}`
    );
    if (!response.ok) throw new Error("Failed to fetch benchmark");
    return await response.json();
  } catch (error) {
    console.error("Error fetching benchmark:", error);
    throw error;
  }
}
