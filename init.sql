-- Database schema for nuts_vision
-- Traces all image processing operations

-- Table: images_input
-- Stores information about uploaded images
CREATE TABLE IF NOT EXISTS images_input (
    image_id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    upload_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    format VARCHAR(10)
);

-- Table: log_jobs
-- Logs detection jobs and their execution details
CREATE TABLE IF NOT EXISTS log_jobs (
    job_id SERIAL PRIMARY KEY,
    image_id INTEGER NOT NULL,
    job_name VARCHAR(255),
    job_folder_path TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    model VARCHAR(255),
    FOREIGN KEY (image_id) REFERENCES images_input(image_id) ON DELETE CASCADE
);

-- Table: detections
-- Stores detection results for each job
CREATE TABLE IF NOT EXISTS detections (
    detection_id SERIAL PRIMARY KEY,
    job_id INTEGER NOT NULL,
    class_name VARCHAR(50) NOT NULL,
    confidence FLOAT NOT NULL,
    bbox_x1 FLOAT NOT NULL,
    bbox_y1 FLOAT NOT NULL,
    bbox_x2 FLOAT NOT NULL,
    bbox_y2 FLOAT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES log_jobs(job_id) ON DELETE CASCADE
);

-- Table: ics_cropped
-- Links jobs to cropped component images
CREATE TABLE IF NOT EXISTS ics_cropped (
    cropped_id SERIAL PRIMARY KEY,
    job_id INTEGER NOT NULL,
    detection_id INTEGER NOT NULL,
    cropped_file_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES log_jobs(job_id) ON DELETE CASCADE,
    FOREIGN KEY (detection_id) REFERENCES detections(detection_id) ON DELETE CASCADE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_log_jobs_image_id ON log_jobs(image_id);
CREATE INDEX IF NOT EXISTS idx_detections_job_id ON detections(job_id);
CREATE INDEX IF NOT EXISTS idx_ics_cropped_job_id ON ics_cropped(job_id);
CREATE INDEX IF NOT EXISTS idx_ics_cropped_detection_id ON ics_cropped(detection_id);
