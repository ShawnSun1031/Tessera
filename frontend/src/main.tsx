import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom";
import { AuthProvider } from "./auth";
import { ProjectShell, RequireAuth } from "./components/Layout";
import { LoginPage, UsersPage } from "./pages/Auth";
import { AnnotatePage, DatasetPage, UploadPage } from "./pages/Data";
import { DatasetPreviewPage } from "./pages/DatasetPreview";
import { ModelPage, ModelsPage, PredictPage } from "./pages/Models";
import { ActivityPage, ProjectsPage, SettingsPage } from "./pages/Projects";
import { JobPage, TrainPage } from "./pages/Train";
import "./styles.css";

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 5_000, refetchOnWindowFocus: true, retry: 1 } },
});

const router = createBrowserRouter([
  { path: "/login", element: <LoginPage /> },
  {
    element: <RequireAuth />,
    children: [
      { path: "/", element: <ProjectsPage /> },
      { path: "/admin/users", element: <UsersPage /> },
      {
        path: "/p/:pid",
        element: <ProjectShell />,
        children: [
          { index: true, element: <Navigate to="upload" replace /> },
          { path: "upload", element: <UploadPage /> },
          { path: "annotate", element: <AnnotatePage /> },
          { path: "dataset", element: <DatasetPage /> },
          { path: "datasets/:datasetId", element: <DatasetPreviewPage /> },
          { path: "train", element: <TrainPage /> },
          { path: "jobs/:jobId", element: <JobPage /> },
          { path: "models", element: <ModelsPage /> },
          { path: "models/:modelId", element: <ModelPage /> },
          { path: "predict", element: <PredictPage /> },
          { path: "activity", element: <ActivityPage /> },
          { path: "settings", element: <SettingsPage /> },
        ],
      },
    ],
  },
  { path: "*", element: <Navigate to="/" replace /> },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <RouterProvider router={router} />
      </AuthProvider>
    </QueryClientProvider>
  </StrictMode>,
);
