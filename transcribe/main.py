#!python3.12
import functools
import logging
import pathlib
from argparse import ArgumentParser

import click
import trio

logger = logging.getLogger(__name__)
minutes_per_part = 10
length = 60 + 17

AUDIO_FILES_PATH = pathlib.Path("~/data/ampotech/5")
WORKING_DIR = "./output"


async def run_and_print(args, *, task_status):
    async with trio.open_nursery() as nursery:
        await nursery.start(
            trio.run_process,
            args,
            capture_stderr=True,
            capture_stdout=True,
            task_status=task_status,
        )


def build_convert_wav_cmd(file: pathlib.Path) -> list[str]:
    return [
        "ffmpeg",
        "-i",
        str(file),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        file.with_suffix(".wav"),
    ]


def build_split_audio_cmd(
    file: pathlib.Path, segment_seconds: int, output_directory: pathlib.Path
) -> list[str]:
    return [
        "ffmpeg",
        "-i",
        str(file),
        "-f",
        "segment",
        "-segment_time",
        str(segment_seconds),
        "-c",
        "copy",
        str(output_directory / str(segment_seconds) / "%03d.m4a"),
    ]


def build_whisper_cpp_cmd(
    binary: pathlib.Path, model: pathlib.Path, file: pathlib.Path
) -> list[str]:
    return [binary, "-m", model, "-f", file, "--output-txt", file.with_suffix(".txt")]


class Namespace:
    binary: pathlib.Path
    model: pathlib.Path
    input_file_path: pathlib.Path
    output_directory: pathlib.Path
    segment_length: int


async def main():
    parser = ArgumentParser()
    parser.add_argument("--binary", "-b", help="whisper.cpp binary", type=pathlib.Path)
    parser.add_argument("--model", "-m", type=pathlib.Path)
    parser.add_argument("--input", "-i", dest="input_file_path", type=pathlib.Path)
    parser.add_argument("--output", "-o", dest="output_directory", type=pathlib.Path)
    parser.add_argument("--segment-length", type=int)
    args = parser.parse_args(namespace=Namespace)

    outputs: list[pathlib.Path] = []
    output_dir = args.output_directory / str(args.segment_length)
    output_dir.mkdir(exist_ok=True, parents=True)

    await trio.run_process(
        build_split_audio_cmd(
            args.input_file_path, args.segment_length, args.output_directory
        )
    )

    all_inputs = sorted(pathlib.Path(output_dir).glob("*.m4a"))
    logger.info("Working on %s", all_inputs)

    for idx, path in enumerate(all_inputs):
        wav_path = path.with_suffix(".wav")
        if not wav_path.exists():
            logger.info("Converting %s...", path)
            await trio.run_process(build_convert_wav_cmd(path))

        logger.info("Transcribing %s...", wav_path := path.with_suffix(".wav"))
        await trio.run_process(build_whisper_cpp_cmd(args.binary, args.model, wav_path))

        # Whisper doesn't care about how we add
        outputs.append(pathlib.Path(f"{wav_path}.txt"))
        logger.info("Done: [%d/%d]", idx + 1, len(all_inputs))

    with pathlib.Path(output_dir / "all.txt").open("w+") as writer:
        for f in sorted(outputs):
            writer.write(f.read_text())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trio.run(main)
