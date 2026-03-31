#ifndef LIBVIDEO_H
#define LIBVIDEO_H

#include "computervison.h"

/*
 * Displays an image as ASCII art in terminal.
 * - `output_width` controls render width in characters.
 * - Height is auto-scaled for terminal character aspect ratio.
 * Returns 0 on success, -1 on invalid input or processing failure.
 */
int libvideo_show_image_ascii(const JCVImage *image, int output_width);

/*
 * Loads an image from disk and displays it as ASCII art in terminal.
 * Returns 0 on success, -1 on failure.
 */
int libvideo_show_image_file_ascii(const char *file_path,
                                   JCVImreadMode mode,
                                   int output_width);

/*
 * Plays a list of image files as an ASCII "video" in terminal.
 * - `fps` must be > 0.
 * - `loops` controls repetitions (>=1).
 * Returns 0 on success, -1 on invalid input or load/display failure.
 */
int libvideo_play_image_sequence_ascii(const char **file_paths,
                                       int frame_count,
                                       JCVImreadMode mode,
                                       int output_width,
                                       int fps,
                                       int loops);

#endif /* LIBVIDEO_H */
