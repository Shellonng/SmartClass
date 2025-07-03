-- Fix assignment_submission_answer table by removing student_id column
-- The student ID is already available through the submission_id

ALTER TABLE `assignment_submission_answer` 
DROP FOREIGN KEY `fk_answer_user`,
DROP INDEX `idx_student_id`,
DROP COLUMN `student_id`; 