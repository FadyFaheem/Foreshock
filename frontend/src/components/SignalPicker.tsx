import type { ChangeEvent } from 'react';
import {
  Button,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Stack,
  Typography,
} from '@mui/material';
import type { SelectChangeEvent } from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import type { Sample } from '../api/foreshock';

interface Props {
  samples: Sample[];
  selectedId: string;
  onSelect: (id: string) => void;
  onUpload: (file: File) => void;
  loading?: boolean;
}

export default function SignalPicker({
  samples,
  selectedId,
  onSelect,
  onUpload,
  loading,
}: Props) {
  const handleFile = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
    e.target.value = '';
  };

  return (
    <Paper sx={{ p: 2 }}>
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        spacing={2}
        alignItems={{ sm: 'center' }}
      >
        <FormControl size="small" sx={{ minWidth: 260 }}>
          <InputLabel id="sample-label">Sample signal</InputLabel>
          <Select
            labelId="sample-label"
            label="Sample signal"
            value={selectedId}
            onChange={(e: SelectChangeEvent) => onSelect(e.target.value)}
          >
            {samples.map((s) => (
              <MenuItem key={s.id} value={s.id}>
                {s.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Button variant="outlined" component="label" startIcon={<UploadFileIcon />}>
          Upload .mat / .csv
          <input hidden type="file" accept=".mat,.csv" onChange={handleFile} />
        </Button>

        {loading && <CircularProgress size={22} />}

        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ display: { xs: 'none', md: 'block' } }}
        >
          Pick a built-in CWRU sample, or upload your own vibration window.
        </Typography>
      </Stack>
    </Paper>
  );
}
